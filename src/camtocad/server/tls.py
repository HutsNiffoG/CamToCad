"""HTTPS op het lokale netwerk (V24, v0.13): een eigen certificaat voor de adressen van deze pc.

Een browser geeft de camera alleen vrij op een beveiligde pagina: https, of http://localhost. Op de
telefoon gaat de pagina naar het LAN-adres van de pc, dus is https nodig voor de live begeleiding.
Een erkende instantie geeft geen certificaat uit voor een adres als 192.168.1.20; daarom maakt de
server zelf een certificaat (zelfondertekend) voor alle LAN-adressen van deze pc en localhost. De
telefoon toont daarbij één keer een waarschuwing ('verbinding is niet privé'). Die accepteer je;
de vingerafdruk in de terminal laat zien dat het certificaat van deze pc komt.

Het certificaat blijft in de datamap staan en wordt hergebruikt zolang het nog een maand geldig is en
alle adressen dekt; anders komt er een nieuw (en dan vraagt de telefoon opnieuw om bevestiging).
"""

from __future__ import annotations

import datetime as dt
import ipaddress
import os
from pathlib import Path

VALID_DAYS = 397  # langer accepteren sommige browsers niet voor een servercertificaat
RENEW_DAYS = 30


def _san(hosts: list[str]):
    from cryptography import x509

    out = []
    for h in dict.fromkeys(hosts):  # volgorde houden, dubbele weg
        try:
            out.append(x509.IPAddress(ipaddress.ip_address(h)))
        except ValueError:
            out.append(x509.DNSName(h))
    return out


def _covers(cert, hosts: list[str]) -> bool:
    from cryptography import x509

    try:
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except x509.ExtensionNotFound:
        return False
    have = {str(v) for v in san.get_values_for_type(x509.IPAddress)} | set(san.get_values_for_type(x509.DNSName))
    return all(h in have for h in hosts)


def fingerprint(cert_path: str | Path) -> str:
    """SHA-256-vingerafdruk van het certificaat (zoals een browser hem toont: AB:CD:...)."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes

    cert = x509.load_pem_x509_certificate(Path(cert_path).read_bytes())
    return cert.fingerprint(hashes.SHA256()).hex(":").upper()


def ensure_certificate(folder: str | Path, hosts: list[str], now: dt.datetime | None = None) -> tuple[Path, Path]:
    """(cert.pem, key.pem) in `folder`, geldig voor `hosts` (IP-adressen en namen). Een bestaand certificaat wordt
    hergebruikt als het alle hosts dekt en nog minstens RENEW_DAYS geldig is."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cert_path, key_path = folder / "cert.pem", folder / "key.pem"
    now = now or dt.datetime.now(dt.timezone.utc)
    if cert_path.exists() and key_path.exists():
        try:
            old = x509.load_pem_x509_certificate(cert_path.read_bytes())
            if old.not_valid_after_utc - now > dt.timedelta(days=RENEW_DAYS) and _covers(old, hosts):
                return cert_path, key_path
        except ValueError:
            pass  # beschadigd: een nieuw maken
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Cam-to-CAD lokaal"),
                      x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Cam-to-CAD")])
    cert = (x509.CertificateBuilder()
            .subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5))
            .not_valid_after(now + dt.timedelta(days=VALID_DAYS))
            .add_extension(x509.SubjectAlternativeName(_san(hosts)), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(x509.KeyUsage(digital_signature=True, key_encipherment=False, content_commitment=False,
                                         data_encipherment=False, key_agreement=True, key_cert_sign=False,
                                         crl_sign=False, encipher_only=False, decipher_only=False), critical=True)
            .sign(key, hashes.SHA256()))
    tmp = key_path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)  # de sleutel alleen leesbaar voor jezelf
    with os.fdopen(fd, "wb") as fh:
        fh.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                   serialization.NoEncryption()))
    tmp.replace(key_path)
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return cert_path, key_path

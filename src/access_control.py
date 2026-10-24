"""Check whether a recognised licence plate is authorised (SQLite database).

Usage (from the repository root):
    python src/access_control.py "MH 20 EE 7602"
    python src/access_control.py --image assets/sample_car.jpg   # detection + OCR + check
"""
import argparse
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = ROOT_DIR / "database" / "alpr.db"


def verifier_acces(plaque, db_path=DB_PATH):
    """Return (autorisé, nom_proprietaire) for a plate number."""
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT nom_proprietaire FROM immatriculations WHERE immatriculation = ?",
            (plaque.strip(),),
        ).fetchone()
    return (row is not None), (row[0] if row else None)


def lire_plaque(image, model=ROOT_DIR / "weights" / "best.pt"):
    """Run detection + OCR on an image and return the first plate read."""
    result = subprocess.run(
        [sys.executable, str(ROOT_DIR / "src" / "predict_with_ocr.py"),
         f"model={model}", f"source={image}"],
        capture_output=True, text=True, cwd=ROOT_DIR,
    )
    found = re.findall(r"OCR Results: \['(.*?)'", result.stdout)
    return found[0] if found else None


def main():
    parser = argparse.ArgumentParser(description="Contrôle d'accès par plaque d'immatriculation")
    parser.add_argument("plaque", nargs="?", help="Numéro de plaque à vérifier")
    parser.add_argument("--image", help="Image d'un véhicule (détection + OCR)")
    args = parser.parse_args()

    plaque = lire_plaque(args.image) if args.image else args.plaque
    if not plaque:
        parser.error("aucune plaque fournie ou détectée")

    autorise, nom = verifier_acces(plaque)
    print(f"Plaque : {plaque}")
    print(f"Accès autorisé ! Bonjour {nom}" if autorise else "Accès refusé")


if __name__ == "__main__":
    main()

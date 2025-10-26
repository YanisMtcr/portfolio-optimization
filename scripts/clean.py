#!/usr/bin/env python
"""
Script to clean all generated files from the portfolio optimization project.
This includes tables, graphs, LaTeX auxiliary files, and compiled PDFs.
"""

import os
import shutil
from pathlib import Path
import glob


def clean_directory(directory_path, description=""):
    """Remove all files in a directory."""
    if os.path.exists(directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"  ✓ Supprimé: {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"  ✓ Supprimé (dossier): {file_path}")
            except Exception as e:
                print(f"  ✗ Erreur lors de la suppression de {file_path}: {e}")
        
        if description:
            print(f"\n✓ {description} nettoyé: {directory_path}")


def clean_latex_files():
    """Remove LaTeX auxiliary and output files."""
    latex_extensions = ['*.aux', '*.log', '*.out', '*.toc', '*.bbl', '*.blg', '*.fls', '*.fdb_latexmk']
    
    # Get the parent directory (project root)
    project_root = Path(__file__).resolve().parents[1]
    
    for ext in latex_extensions:
        for file_path in glob.glob(str(project_root / ext)):
            try:
                os.remove(file_path)
                print(f"  ✓ Supprimé: {file_path}")
            except Exception as e:
                print(f"  ✗ Erreur: {file_path}: {e}")
    
    print("\n✓ Fichiers LaTeX auxiliaires nettoyés")


def main():
    print("=" * 60)
    print("Nettoyage des fichiers générés")
    print("=" * 60)
    
    # Get the project root directory
    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)
    
    # Clean results directory
    print("\n1. Nettoyage du dossier 'results'...")
    results_dir = os.path.join(project_root, "results")
    clean_directory(results_dir, "Fichiers de résultats")
    
    # Clean config directory
    print("\n2. Nettoyage du dossier 'config'...")
    config_dir = os.path.join(project_root, "config")
    clean_directory(config_dir, "Fichiers de configuration/graphiques")
    
    # Clean LaTeX auxiliary files
    print("\n3. Nettoyage des fichiers LaTeX auxiliaires...")
    clean_latex_files()
    
    # Clean main.tex output (if exists)
    print("\n4. Nettoyage du PDF généré...")
    main_pdf = os.path.join(project_root, "main.pdf")
    if os.path.exists(main_pdf):
        try:
            os.remove(main_pdf)
            print(f"  ✓ Supprimé: {main_pdf}")
        except Exception as e:
            print(f"  ✗ Erreur: {main_pdf}: {e}")
    
    print("\n" + "=" * 60)
    print("Nettoyage terminé !")
    print("=" * 60)


if __name__ == "__main__":
    main()

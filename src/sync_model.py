"""
Módulo para Download de Modelos do Google Drive.

Este script utiliza a biblioteca 'gdown' para baixar um arquivo a partir de
um ID de compartilhamento do Google Drive, facilitando a sincronização de
modelos treinados em ambientes externos (como o Google Colab) com o projeto local.

Autor: GitHub Copilot
Data: 2026-03-05
"""
import argparse
import gdown
from pathlib import Path

def download_model_from_gdrive(file_id: str, output_path: str):
    """
    Baixa um arquivo do Google Drive usando seu ID de compartilhamento.

    Args:
        file_id (str): O ID do arquivo no Google Drive.
        output_path (str): O caminho local onde o arquivo será salvo.
    """
    output_path = Path(output_path)
    # Garante que o diretório de destino exista
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Iniciando download do Google Drive (ID: {file_id})...")
    print(f"Salvando em: {output_path}")
    
    try:
        # Tenta baixar usando o ID do arquivo
        gdown.download(id=file_id, output=str(output_path), quiet=False, fuzzy=True)
        
        if output_path.exists():
            print(f"\nModelo salvo com sucesso em: {output_path}")
        else:
            print("\nERRO: O download parece ter falhado, pois o arquivo não foi criado.")
            print("Verifique o ID do arquivo e as permissões de compartilhamento.")

    except Exception as e:
        print(f"\nOcorreu um erro durante o download: {e}")
        print("DICA: Certifique-se de que o arquivo no Google Drive está com o 'Acesso geral' definido como 'Qualquer pessoa com o link'.")

def main():
    """Função principal para parsear argumentos e iniciar o download."""
    parser = argparse.ArgumentParser(
        description="Script para baixar modelos do Google Drive.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "file_id",
        help="O ID do arquivo no Google Drive.\n"
             "Você pode extrair o ID do link de compartilhamento. Exemplo:\n"
             "No link 'https://drive.google.com/file/d/1aBcDeFgHiJkLmNoPqRsTuVwXyZ/view?usp=sharing',\n"
             "O ID é '1aBcDeFgHiJkLmNoPqRsTuVwXyZ'."
    )
    parser.add_argument(
        "output_path",
        help="Caminho completo onde o modelo será salvo (ex: models/yolo11x-pose.pt)."
    )
    args = parser.parse_args()

    download_model_from_gdrive(args.file_id, args.output_path)

if __name__ == "__main__":
    main()

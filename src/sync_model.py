import argparse
import gdown
from pathlib import Path

def sync_folder_from_gdrive(folder_id: str, output_parent_dir: str):
    """
    Baixa uma pasta do Google Drive para um diretório local.

    Args:
        folder_id (str): O ID da pasta no Google Drive.
        output_parent_dir (str): O diretório local onde a pasta do Drive será salva.
    """
    output_path = Path(output_parent_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Iniciando sincronização da pasta do Google Drive (ID: {folder_id})...")
    print(f"A pasta do Drive será salva dentro de: {output_path.resolve()}")
    
    try:
        # Baixa a pasta. 'remaining_ok=True' pula arquivos que já existem,
        # agindo como uma sincronização. Se a pasta de destino já existir,
        # o conteúdo será mesclado/atualizado.
        gdown.download_folder(
            id=folder_id,
            output=str(output_path),
            quiet=False,
            remaining_ok=True,
            use_cookies=False
        )
        
        print(f"\nDiretório sincronizado com sucesso!")

    except Exception as e:
        print(f"\nOcorreu um erro durante a sincronização: {e}")
        print("DICA: Certifique-se de que a pasta no Google Drive e todos os seus arquivos estão com o 'Acesso geral' definido como 'Qualquer pessoa com o link'.")


def main():
    """Função principal para parsear argumentos e iniciar o download."""
    parser = argparse.ArgumentParser(
        description="Script para sincronizar um diretório inteiro do Google Drive.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "folder_id",
        help="O ID da PASTA no Google Drive.\n"
             "Você pode extrair o ID do link de compartilhamento da pasta. Exemplo:\n"
             "No link 'https://drive.google.com/drive/folders/1AeP65jbcZzdakLDhyYDAVgyR5NMnS9wp?usp=sharing',\n"
             "O ID é '1AeP65jbcZzdakLDhyYDAVgyR5NMnS9wp'."
    )
    parser.add_argument(
        "output_parent_dir",
        nargs='?',
        default='.',
        help="Diretório PAI onde a pasta do Drive será salva. \n"
             "Se a pasta no Drive se chama 'models' e este argumento for '.', a pasta será salva em './models'.\n"
             "(Padrão: diretório atual './')"
    )
    args = parser.parse_args()
    
    sync_folder_from_gdrive(args.folder_id, args.output_parent_dir)

if __name__ == "__main__":
    main()

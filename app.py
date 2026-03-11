"""
Entrypoint para Hugging Face Spaces.
Sobe a API FastAPI em background e o Streamlit em foreground.
"""
import subprocess
import time
import sys
import os
import signal

def kill_process_on_port(port):
    """Mata qualquer processo rodando na porta especificada."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    print(f"🔄 Processo {pid} na porta {port} foi terminado")
                except ProcessLookupError:
                    pass
            time.sleep(2)
    except FileNotFoundError:
        # lsof não disponível, ignorar
        pass

def main():
    print("🚀 Iniciando Cow Classifier no Hugging Face Spaces...")
    
    # Limpar porta 8000 se estiver ocupada
    kill_process_on_port(8000)
    
    # Subir API FastAPI em background
    print("📡 Subindo API FastAPI na porta 8000...")
    api_process = subprocess.Popen(
        ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Aguardar API inicializar
    time.sleep(8)
    
    # Verificar se API está rodando
    if api_process.poll() is not None:
        print("❌ Erro ao iniciar API")
        stdout, stderr = api_process.communicate()
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
        sys.exit(1)
    
    print("✅ API rodando em http://localhost:8000")
    print("🎨 Subindo Streamlit na porta 7860...")
    
    # Definir variável de ambiente para o Streamlit saber onde está a API
    os.environ["API_URL"] = "http://127.0.0.1:8000"
    
    # Subir Streamlit (foreground) - porta 7860 é padrão do HF Spaces
    subprocess.run([
        "streamlit", "run", "src/ui/streamlit_app.py",
        "--server.port", "7860",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
        "--server.maxUploadSize", "200"
    ])

if __name__ == "__main__":
    main()

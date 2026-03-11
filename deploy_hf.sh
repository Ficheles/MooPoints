#!/bin/bash
set -e

# Carregar configurações do .env
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

echo "🔐 Fazendo login no Hugging Face..."

# Verificar se HF_TOKEN está definido
if [ -z "$HF_TOKEN" ]; then
    echo "Por favor, obtenha seu token em: https://huggingface.co/settings/tokens"
    huggingface-cli auth login
else
    huggingface-cli auth login --token "$HF_TOKEN"
fi

echo ""
echo "📝 Configurando o Space..."

# Usar variáveis do .env ou solicitar input
if [ -z "$HF_USERNAME" ]; then
    read -p "Digite seu username do Hugging Face: " HF_USERNAME
fi

if [ -z "$HF_SPACE_NAME" ]; then
    read -p "Digite o nome do Space (ex: cow-classifier): " HF_SPACE_NAME
fi

SPACE_URL="https://huggingface.co/spaces/${HF_USERNAME}/${HF_SPACE_NAME}"

echo ""
echo "🎯 Criando Space: ${SPACE_URL}"
huggingface-cli repo create "${HF_SPACE_NAME}" --type space --space-sdk docker || echo "⚠️  Space já existe, continuando..."

echo ""
echo "🔗 Adicionando remote do Hugging Face..."
git remote remove hf 2>/dev/null || true
git remote add hf "https://huggingface.co/spaces/${HF_USERNAME}/${HF_SPACE_NAME}"

echo ""
echo "📦 Preparando arquivos para deploy..."

# Backup do README original e usar o README do Space
cp README.md README_GITHUB.md
cp README_SPACES.md README.md

# Usar requirements específico para HF
cp requirements.txt requirements_local.txt
cp requirements_hf.txt requirements.txt

echo ""
echo "💾 Commitando alterações para deploy..."
git add .
git commit -m "deploy: configure for Hugging Face Spaces" || echo "Nada para commitar"

echo ""
echo "🚀 Fazendo push para o Hugging Face Spaces..."
git push hf deploy/huggingface-spaces:main --force

echo ""
echo "✅ Deploy concluído!"
echo "🌐 Seu Space estará disponível em: ${SPACE_URL}"
echo ""
echo "⏱️  Aguarde alguns minutos para o build e inicialização."
echo "📊 Você pode acompanhar em: ${SPACE_URL}/settings"

# Restaurar arquivos locais
echo ""
echo "🔄 Restaurando arquivos locais..."
git reset --soft HEAD~1
git restore --staged .
mv README_GITHUB.md README.md 2>/dev/null || true
mv requirements_local.txt requirements.txt 2>/dev/null || true

echo ""
echo "✨ Pronto! Seu repositório local permanece inalterado."

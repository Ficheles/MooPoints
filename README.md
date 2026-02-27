# Cow Classifier - Ambiente Conda

Este projeto utiliza um ambiente Conda para garantir a reprodutibilidade e facilitar a instalação das dependências.

## 1. Pré-requisitos

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) ou [Anaconda](https://www.anaconda.com/products/distribution) instalado

## 2. Criando o ambiente Conda

Você pode criar um novo ambiente Conda com Python 3.12 (ou a versão desejada) usando o comando:

```bash
conda create -n cow-classifier python=3.12
```

Ative o ambiente:

```bash
conda activate cow-classifier
```

## 3. Instalando as dependências

### Usando requirements-conda.txt

Se preferir instalar as dependências listadas em `requirements-conda.txt`:

```bash
conda install --yes --file requirements-conda.txt
```

## 4. Executando o script de organização

Após instalar as dependências, execute:

```bash
python src/prepare_dataset.py

## Deve depois que organizar as fotos, criar os links simbólicos e os folds, execute:
python src/validate_annotations.py
```

Pode também usar o script `run.sh` para executar ambos os passos de preparação e validação:

```bash
./run.sh
```

Isso irá organizar as fotos e criar os folds para o projeto.

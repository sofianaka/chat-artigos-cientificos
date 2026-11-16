
# em vetores e achar os trechos mais parecidos com a pergunta.

import numpy as np
from fastembed import TextEmbedding
from pypdf import PdfReader

# Modelo que transforma texto em vetor. É multilíngue: uma pergunta em
# português encontra trechos de um artigo em inglês.
MODELO_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TAMANHO_TRECHO = 1000   # quantos caracteres tem cada trecho
SOBREPOSICAO = 200      # quanto um trecho repete do anterior, pra não cortar ideias no meio
QUANTOS_TRECHOS = 4     # quantos trechos vão pro LLM em cada pergunta
LIMITE_PAGINAS = 60     # artigos maiores que isso são cortados


def carregar_modelo():
    """Carrega o modelo de embeddings (na primeira vez ele é baixado)."""
    return TextEmbedding(MODELO_EMBEDDING)


def ler_pdf(arquivo):
    """Devolve uma lista com o texto de cada página: [(1, "texto..."), (2, "texto..."), ...]"""
    leitor = PdfReader(arquivo)
    paginas = []
    for numero, pagina in enumerate(leitor.pages[:LIMITE_PAGINAS], start=1):
        texto = pagina.extract_text() or ""
        texto = " ".join(texto.split())  # tira quebras de linha e espaços sobrando
        if texto:
            paginas.append((numero, texto))
    return paginas


def dividir_em_trechos(paginas):
    """Corta o texto de cada página em pedaços menores, guardando de que página veio."""
    trechos = []
    passo = TAMANHO_TRECHO - SOBREPOSICAO
    for numero, texto in paginas:
        for inicio in range(0, len(texto), passo):
            pedaco = texto[inicio : inicio + TAMANHO_TRECHO]
            if len(pedaco) > 50:  # ignora restinhos muito curtos
                trechos.append({"pagina": numero, "texto": pedaco})
    return trechos


def criar_vetores(modelo, textos):
    """Transforma cada texto num vetor de números (embedding)."""
    vetores = np.array(list(modelo.embed(textos)))
    # Deixa todos com tamanho 1, assim o produto entre dois vetores
    # já é a similaridade de cosseno
    return vetores / np.linalg.norm(vetores, axis=1, keepdims=True)


def buscar(modelo, pergunta, trechos, vetores):
    """Devolve os trechos mais parecidos com a pergunta, do mais parecido pro menos."""
    vetor_pergunta = criar_vetores(modelo, [pergunta])[0]
    similaridades = vetores @ vetor_pergunta
    melhores = np.argsort(similaridades)[::-1][:QUANTOS_TRECHOS]
    return [trechos[i] for i in melhores]

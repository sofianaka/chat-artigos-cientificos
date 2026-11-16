
import os

import streamlit as st

import rag
from llm import responder

PERGUNTAS_SUGERIDAS = [
    "Qual é o objetivo do artigo?",
    "Qual foi a metodologia usada?",
    "Quais foram os resultados?",
]

st.set_page_config(page_title="Chat com Artigos Científicos", page_icon="📚", layout="wide")


def ler_arquivo(caminho):
    """Lê um arquivo de texto (CSS ou HTML) e devolve o conteúdo."""
    with open(caminho, encoding="utf-8") as arquivo:
        return arquivo.read()


@st.cache_resource(show_spinner="Carregando o modelo de busca...")
def modelo_de_busca():
    return rag.carregar_modelo()


def processar_artigo(pdf):
    """Lê o PDF e guarda os trechos e vetores. Só roda quando o artigo muda."""
    identificador = f"{pdf.name}-{pdf.size}"
    if st.session_state.get("artigo") == identificador:
        return

    with st.spinner("Lendo o artigo..."):
        paginas = rag.ler_pdf(pdf)
        trechos = rag.dividir_em_trechos(paginas)
        vetores = None
        if trechos:
            vetores = rag.criar_vetores(modelo_de_busca(), [t["texto"] for t in trechos])

    st.session_state["artigo"] = identificador
    st.session_state["paginas"] = len(paginas)
    st.session_state["trechos"] = trechos
    st.session_state["vetores"] = vetores
    st.session_state["mensagens"] = []  # artigo novo, conversa nova


def gerar_resposta(pergunta):
    """Busca os trechos mais parecidos e pede a resposta pro LLM."""
    if "GROQ_API_KEY" not in os.environ:
        aviso = (
            'Faltou a chave da API. No terminal, rode export GROQ_API_KEY="sua-chave" '
            "e abra o app de novo."
        )
        return aviso, []
    try:
        with st.spinner("Procurando no artigo..."):
            usados = rag.buscar(
                modelo_de_busca(),
                pergunta,
                st.session_state["trechos"],
                st.session_state["vetores"],
            )
            return responder(pergunta, usados), usados
    except Exception as erro:
        return f"Não foi possível responder agora. Detalhe: {erro}", []


def mostrar_chat():
    mensagens = st.session_state["mensagens"]
    area_da_conversa = st.container()

    pergunta = None
    colunas = st.columns(len(PERGUNTAS_SUGERIDAS))
    for coluna, sugestao in zip(colunas, PERGUNTAS_SUGERIDAS):
        if coluna.button(sugestao, width="stretch"):
            pergunta = sugestao

    pergunta = st.chat_input("Escreva sua pergunta") or pergunta

    if pergunta:
        mensagens.append({"quem": "user", "texto": pergunta})
        resposta, usados = gerar_resposta(pergunta)
        mensagens.append({"quem": "assistant", "texto": resposta, "trechos": usados})

    # Mostra a conversa inteira, da primeira mensagem até a última
    with area_da_conversa:
        for mensagem in mensagens:
            with st.chat_message(mensagem["quem"]):
                st.write(mensagem["texto"])
                if mensagem.get("trechos"):
                    with st.expander("Trechos do artigo usados na resposta"):
                        for trecho in mensagem["trechos"]:
                            st.caption(f"Página {trecho['pagina']}")
                            st.write(trecho["texto"])


# ---------- Visual e banner do topo ----------
st.markdown(f"<style>{ler_arquivo('style.css')}</style>", unsafe_allow_html=True)
st.markdown(ler_arquivo("templates/cabecalho.html"), unsafe_allow_html=True)

# ---------- Duas colunas: artigo à esquerda, conversa à direita ----------
coluna_artigo, coluna_chat = st.columns([1, 2], gap="large")

with coluna_artigo:
    with st.container(key="cartao_artigo"):
        st.subheader("1. Envie o artigo")
        pdf = st.file_uploader("Arquivo em PDF", type="pdf")
        if pdf:
            processar_artigo(pdf)
            st.caption(
                f"{st.session_state['paginas']} páginas lidas · "
                f"{len(st.session_state['trechos'])} trechos indexados"
            )
    st.markdown(ler_arquivo("templates/como_funciona.html"), unsafe_allow_html=True)

with coluna_chat:
    with st.container(key="cartao_chat"):
        st.subheader("2. Pergunte sobre ele")
        if pdf is None:
            st.info("Envie um artigo em PDF pra começar a conversa.")
        elif not st.session_state["trechos"]:
            st.warning(
                "Não consegui extrair texto desse PDF. Ele pode ser um arquivo "
                "escaneado (imagem). Tente outro artigo."
            )
        else:
            mostrar_chat()

st.markdown(ler_arquivo("templates/rodape.html"), unsafe_allow_html=True)

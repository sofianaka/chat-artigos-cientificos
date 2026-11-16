# A parte  do RAG monta o prompt com os trechos encontrados
# e pede a resposta pro LLM.

import os

from groq import Groq


MODELO = "openai/gpt-oss-120b"

PROMPT = """Você responde perguntas sobre um artigo científico.

Regras:
- Use SOMENTE os trechos do artigo que vêm junto com a pergunta.
- Se a resposta não estiver nos trechos, diga que não encontrou essa informação no artigo. Não invente.
- Cite a página de onde tirou cada informação, no formato (p. 3).
- Responda em português do Brasil, de forma clara e direta, mesmo que o artigo esteja em outro idioma."""


def responder(pergunta: str, trechos: list) -> str:
    """Manda a pergunta e os trechos pro LLM e devolve o texto da resposta."""
    cliente = Groq(api_key=os.environ["GROQ_API_KEY"])

    # Junta os trechos num texto só
    contexto = "\n\n".join(f"[Página {t['pagina']}]\n{t['texto']}" for t in trechos)

    resposta = cliente.chat.completions.create(
        model=MODELO,
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": f"Trechos do artigo:\n\n{contexto}\n\nPergunta: {pergunta}"},
        ],
        temperature=0,
    )
    return resposta.choices[0].message.content

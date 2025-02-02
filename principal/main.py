import tempfile

import streamlit as st # Biblioteca para rodar o código localmente

from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

import requests
import time
from loaders import *

TIPOS_ARQUIVOS_VALIDOS = [
    'Site',
    'Youtube',
    'PDF',
    'CSV',
    'TXT'
] # Organização do tipo dos arquivos a serem selecionados

def call_deepseek(api_key, model, messages): # Função para chamar o DeepSeek
    url = "https://api.deepseek.com/v1/chat/completions"  # DeepSeek API endpoint
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": messages
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API Request Failed: {e}")
        return None

CONFIG_MODELOS = {
    'Groq': {
        'modelos': ['llama-3.3-70b-versatile', 'gemma2-9b-it'],
         'chat': ChatGroq},
    'OpenAI': {
        'modelos': ['gpt-4o-mini', 'gpt-4o'],
           'chat': ChatOpenAI},
    'DeepSeek': {
        'modelos': ['deepseek_chat', 'deepseek-v3'],
         'chat': lambda api_key: api_key}
} # Organização do tipo de APIs a serem selecionadas

MEMORIA = ConversationBufferMemory()

def carrega_arquivo(tipo_arquivo, arquivo):
    if tipo_arquivo == 'Site':
        documento = carrega_site(arquivo)
    elif tipo_arquivo == 'Youtube':
        documento = carrega_youtube(arquivo)
    elif tipo_arquivo == 'PDF':
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_pdf(nome_temp)
    elif tipo_arquivo == 'CSV':
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_csv(nome_temp)
    elif tipo_arquivo == 'TXT':
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_txt(nome_temp)
    else:
        documento = ''
    return documento

def carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo, chain=None):
    if arquivo is None:
        st.error('Por favor, carregue um arquivo.')
        return

    documento = carrega_arquivo(tipo_arquivo, arquivo)

    system_message = '''Você é um assistente amigável chamado Oráculo.
    Você possui acesso às seguintes informações vindas 
    de um documento {}: 

    ####
    {}
    ####

    Utilize as informações fornecidas para basear as suas respostas.

    Sempre que houver $ na sua saída, substita por S.

    Se a informação do documento for algo como "Just a moment...Enable JavaScript and cookies to continue" 
    sugira ao usuário carregar novamente o Oráculo!'''.format(tipo_arquivo, documento)
    template = ChatPromptTemplate.from_messages([
        ('system', system_message),
        ('placeholder', '{chat_history}'),
        ('user', '{input}')
    ])

    if provedor not in CONFIG_MODELOS:
        raise ValueError("Provedor inválido")

    config = CONFIG_MODELOS[provedor]
    if modelo not in config['modelos']:
        raise ValueError("Modelos de provedores inválidos")

    if provedor == 'DeepSeek':
        st.session_state['deepseek_api_key'] = api_key
        st.session_state['deepseek_model'] = modelo
        st.session_state['chain'] = None
    else:
        chat = CONFIG_MODELOS[provedor]['chat'](model=modelo, api_key=api_key)
        chain = template | chat

    st.session_state['chain'] = chain
    st.session_state['model'] = modelo
    st.session_state['provider'] = provedor
    st.session_state['api_key'] = api_key
    st.success(f"Modelo {modelo} de {provedor} carregado com sucesso!")

def pagina_chat():
    st.header('Bem-Vindo ao Oráculo 🔮', divider=True) # Título

    chain = st.session_state.get('chain')
    provedor = st.session_state.get('provider')
    memoria = st.session_state.get('memoria', MEMORIA) # As menssagens que são inicializada no começo

    #Display chat histórico
    for menssagem in memoria.buffer_as_messages: # Para cada nova mensagem é incrementado um iten na variavel chat
        chat = st.chat_message(menssagem.type) # Chat recebe mensagem na posição 0
        chat.markdown(menssagem.content)

    # Input do usuário
    input_usuario = st.chat_input('Fale com o oráculo') # Barra de conversa
    if input_usuario: # Esse é o input do usuário
        chat = st.chat_message('human')
        chat.markdown(input_usuario)

        if provedor == 'DeepSeek':
            api_key = st.session_state.get('deepseek_api_key')
            modelo = st.session_state.get('deepseek_model')
            messages = [{"role": "user", "content": input_usuario}]
            response = call_deepseek(api_key, modelo, messages)
            if response is None:
                st.error("Falha em obter uma resposta do DeepSeek. Por favor, cheque sua chave API e tente novamente.")
                return

            try:
                resposta = response['choices'][0]['message']['content']
            except KeyError:
                st.error("Falha em extrair uma resposta do DeepSeek API. Por favor cheque a estrutura de resposta API.")
                print("Full API Response:", response)
                resposta = "Desculpe, ocorreu um erro ao processar sua solicitação."
        else:
            chat = st.chat_message('ai')
            resposta = chat.write_stream(chain.stream({
                'input': input_usuario,
                'chat_history': memoria.buffer_as_messages}))

        memoria.chat_memory.add_user_message(input_usuario)
        memoria.chat_memory.add_ai_message(resposta)
        st.session_state['memoria'] = memoria # Mostra o input do usuário

def sidebar():
    tabs = st.tabs(['Upload de Arquivos', 'Seleção de Modelos']) # Escolhas na parte esquerda da tela [0] ou [1]
    with tabs[0]: # Aba [0]
        tipo_arquivo = st.selectbox('Selecione o tipo de arquivo', TIPOS_ARQUIVOS_VALIDOS) # Vamos escolher o tipo do arquivo por um selecionador

        if tipo_arquivo == 'Site': # Selecionando Site
            arquivo = st.text_input('Digite a URL do site: ') # Inputando a URL
        elif tipo_arquivo == 'Youtube': # Selecionando Video
            arquivo = st.text_input('Digite apenas o código final do vídeo ex:"OWBT5EEikj8": ') # Inputando a url do video
        elif tipo_arquivo == 'PDF': # Selecionando o arquivo PDF
            arquivo = st.file_uploader('Faça o upload do arquivo pdf', type=['.pdf']) # Fazendo o upload em .pdf
        elif tipo_arquivo == 'CSV': # Selecionando o arquivo CSV
            arquivo = st.file_uploader('Faça o upload do arquivo pdf', type=['.csv']) # Fazendo o upload em .cvs
        elif tipo_arquivo == 'TXT': # Selecionando o arquivo TXT
            arquivo = st.file_uploader('Faça o upload do arquivo pdf', type=['.txt']) # Fazendo o upload em .txt
        else:
            arquivo = None
    with tabs[1]: # Selecionando o modelo de API
        provedor = st.selectbox('Selecione o provedor dos modelos', CONFIG_MODELOS.keys()) # Uma caixa de seleção para escolha da API já organizada em um dicionário
        modelo = st.selectbox('Selecione o modelo', CONFIG_MODELOS[provedor]['modelos']) # Uma caixa de seleção para escolha do modelo da API selecionada

        api_key = st.text_input(
            f'Digite a API do modelo: {provedor}',
            type="password" # Escode a API
        )

        st.session_state[f'api_key_{provedor}'] = api_key # Inicialização com a respectiva API_KEY

    if st.button('Inicializar o Oráculo', use_container_width=True):
        carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)
    if st.button('Apagar Histórico de Conversa', use_container_width=True):
        st.session_state['memoria'] = MEMORIA

def main():
    with st.sidebar:
        sidebar()
    pagina_chat()

if __name__ == '__main__':
    main()
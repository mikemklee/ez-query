from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from llama_index import GPTKeywordTableIndex, LLMPredictor, download_loader


def main():
    load_dotenv()

    # fetch data from Zendesk
    ZendeskReader = download_loader("ZendeskReader")
    loader = ZendeskReader(zendesk_subdomain="growsumo", locale="en-us")
    documents = loader.load_data()

    # setup model and create index
    llm_predictor = LLMPredictor(llm=ChatOpenAI(model_name="gpt-3.5-turbo"))
    index = GPTKeywordTableIndex(documents, llm_predictor=llm_predictor)

    # run query
    response = index.query("What are referrals?")
    print(response)


if __name__ == "__main__":
    main()

import argparse
import os
import pickle
import time

from dotenv import load_dotenv
from llama_index.optimization.optimizer import SentenceEmbeddingOptimizer
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from llama_index import (
    GPTKeywordTableIndex,
    GPTSimpleKeywordTableIndex,
    LLMPredictor,
    download_loader,
)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Query zendesk articles",
    )

    parser.add_argument(
        "--subdomain",
        type=str,
        help="Zendesk subdomain to fetch articles from",
    )

    parser.add_argument(
        "--question",
        type=str,
        help="What you want to find out from the articles",
    )

    args = parser.parse_args()

    start_time = time.time()

    documents_file_name = f"{args.subdomain}.documents.pickle"
    index_file_name = f"{args.subdomain}.index.json"

    print("\nStep 1: Grabbing zendesk articles")
    # Check if the documents are available in a local file
    if os.path.exists(documents_file_name):
        print("  - found saved documents, loading from there")
        with open(documents_file_name, "rb") as f:
            documents = pickle.load(f)
    else:
        # fetch data from Zendesk
        print("  - no saved documents, fetching from Zendesk")
        ZendeskReader = download_loader("ZendeskReader")
        loader = ZendeskReader(
            zendesk_subdomain=args.subdomain,
            locale="en-us",
        )
        documents = loader.load_data()

        # Save the documents to a local file
        print("  - saving documents to local file")
        with open(documents_file_name, "wb") as f:
            pickle.dump(documents, f)

    # setup model
    print("\nStep 2: Setting up LLM")

    llm_predictor = LLMPredictor(llm=ChatOpenAI(model_name="gpt-3.5-turbo"))

    # OpenAI keeps complaining that `gpt-3.5-turbo` is "busy"
    # use `text-davinci-003` instead for now
    # caveat: `text-davinci-003` is MUCH more expensive than `gpt-3.5-turbo`
    # llm_predictor = LLMPredictor(llm=OpenAI(model_name="text-davinci-003"))

    print("\nStep 3: Setting up index")
    # Check if the index is available in a local file
    if os.path.exists(index_file_name):
        print("  - found saved index, loading from there")
        index = GPTSimpleKeywordTableIndex.load_from_disk(index_file_name)
    else:
        # setup index
        print("  - no saved index, building one from documents")
        index = GPTSimpleKeywordTableIndex(
            documents=documents,
            llm_predictor=llm_predictor,
        )

        # Save the index to a local file
        print("  - saving index to local file")
        index.save_to_disk(index_file_name)

    # run query
    print("\nStep 4: Running query")
    response = index.query(
        args.question,
        optimizer=SentenceEmbeddingOptimizer(
            percentile_cutoff=0.75,
        ),
    )

    # print results
    print("\nQUESTION:", args.question)
    print("\nRESPONSE:", response)

    end_time = time.time()
    print("\nTotal duration:", end_time - start_time, "seconds")


if __name__ == "__main__":
    main()

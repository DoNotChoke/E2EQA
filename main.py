import streamlit as st
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, pipeline
import faiss

# Kiểm tra thiết bị
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load dataset
DATASET_NAME = 'squad_v2'
raw_datasets = load_dataset(DATASET_NAME, split="train+validation").shard(num_shards=40, index=0)

# Lọc những câu hỏi có câu trả lời
raw_datasets = raw_datasets.filter(lambda x: len(x["answers"]["text"]) > 0)

# Giữ lại các cột quan trọng
columns_to_keep = ['id', 'context', 'question', 'answers']
raw_datasets = raw_datasets.remove_columns(set(raw_datasets.column_names) - set(columns_to_keep))

# Load tokenizer và model
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)

# Hàm tạo embedding
def get_embeddings(text_list):
    encoded_input = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return model_output.last_hidden_state[:, 0]

# Tạo embeddings cho tập dữ liệu
EMBEDDING_COLUMN = "question_embedding"
embedding_dataset = raw_datasets.map(
    lambda x: {EMBEDDING_COLUMN: get_embeddings(x["question"]).detach().cpu().numpy()[0]}
)
embedding_dataset.add_faiss_index(column=EMBEDDING_COLUMN)

# Load pipeline trả lời câu hỏi
PIPELINE_NAME = "question-answering"
QA_MODEL_NAME = "DoNotChoke/distilbert-finetuned-squadv2"
qa_pipeline = pipeline(PIPELINE_NAME, model=QA_MODEL_NAME)

# Giao diện Streamlit
st.title("Question Answering System")
st.write("Nhập câu hỏi của bạn và hệ thống sẽ tìm kiếm câu trả lời phù hợp.")

# Ô nhập câu hỏi
user_question = st.text_input("Nhập câu hỏi:", "When did Beyonce start becoming popular?")

if st.button("Tìm kiếm câu trả lời"):
    if user_question:
        input_question_embedding = get_embeddings([user_question]).cpu().detach().numpy()
        TOP_K = 5
        scores, samples = embedding_dataset.get_nearest_examples(EMBEDDING_COLUMN, input_question_embedding, k=TOP_K)

        st.subheader("Kết quả tìm kiếm:")
        for idx, score in enumerate(scores):
            context = samples["context"][idx]
            answer = qa_pipeline(question=user_question, context=context)

            st.write(f"**Top {idx + 1} (Score: {score:.4f})**")
            st.write(f"**Context:** {context}")
            st.write(f"**Answer:** {answer['answer']}")
            st.write("---")
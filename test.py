from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """Artificial Intelligence is transforming industries. Businesses use AI for automation, data analysis, and customer support.
With advancements in deep learning, AI systems can now process images, speech, and text efficiently."""

# Create a text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)

chunks = splitter.split_text(text)
print(chunks)

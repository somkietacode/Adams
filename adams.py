from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from nltk.tokenize import sent_tokenize as sentence_tokenize
import warnings, torch, requests
from model_saver import ModelSaver
from bs4 import BeautifulSoup

warnings.filterwarnings('ignore')


def google_search(user_input):
    find = False
    query = user_input
    url = f"https://www.google.com/search?q={query}&hl=en"
    # send a GET request to the Google search page
    response = requests.get(url)

    # parse the response using beautifulsoup
    soup = BeautifulSoup(response.text, "html.parser")
    # extract the search results
    results = soup.select("a")
    find = False
    Text = ""
    for result in results:
        url = result.get("href")
        if "/url?q=https://" in url and "google" not in url :
            url = "https://www.google.com"+url
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            elements = soup.find_all(['h1','h2','h3','h4', 'p'])
            elements_with_index = [(i, element) for i, element in enumerate(elements)]
            sorted_elements_with_index = sorted(elements_with_index, key=lambda x: x[0])
            sorted_elements = [element for i, element in sorted_elements_with_index]
            for element in sorted_elements:
                Text += element.text+"\n"
            return Text

if __name__ == '__main__':
    #print(Chatbot.training_data)
    print("\n")
    saver = ModelSaver()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('models/adams', pad_token_id=tokenizer.eos_token_id)
    while True :

        q = input("User > ")
        Text = google_search(q)

        with open("data/data.txt","w", encoding="utf-8") as data:
            data.write(Text)
            data.close()
        train_path = 'data/data.txt'

        # Create a TextDataset object from the training data
        dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=train_path,
            block_size=128
        )

        # Create a DataCollatorForLanguageModeling object
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        )

        # Define the training arguments
        training_args = TrainingArguments(
            output_dir='results',
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=16,
            save_steps=10_000,
            save_total_limit=2,
            prediction_loss_only=True,
            logging_steps=100,
            logging_first_step=True,
            learning_rate=1e-4
        )

        # Create a Trainer object
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator
        )

        # Fine-tune the pre-trained model on your training data
        trainer.train()
        trainer.save_model('models/adams')
        saver.save_model(model,"adam")
        prompt = q
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        attention_mask = torch.ones_like(input_ids)
        attention_mask[input_ids == 0] = 0
        output = model.generate(input_ids,attention_mask=attention_mask, max_length=100,num_beams=5,temperature=0.8,no_repeat_ngram_size=2,num_return_sequences=3, do_sample=True)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        tokenized_sentences = sentence_tokenize(response)
        tokenized_sentences.remove(tokenized_sentences[-1])
        for i in tokenized_sentences :
            print("Bot > "+i)

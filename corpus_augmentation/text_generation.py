from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelWithLMHead
from transformers import pipeline
import random
from disambiguation_methods.word2vec import generate_additional_data, load_keyed_vector_model
from util import load
from disambiguation_methods.LSTM import preprocess_tokens


def sloberta_generation(n_masks, masked_sentence, seed_sentence):
    tokenizer = AutoTokenizer.from_pretrained("EMBEDDIA/sloberta")
    model = AutoModelForMaskedLM.from_pretrained("EMBEDDIA/sloberta")
    classifier = pipeline("fill-mask", model=model, tokenizer=tokenizer)

    if n_masks == 1:
        return classifier(masked_sentence, top_k=5)
    else:
        model.eval()
        seed = seed_sentence
        n = n_masks
        output = [seed]
        for _ in range(n):
            new_out = []
            for item in output:
                masked_input = item + " <mask>"
                res = classifier(masked_input, top_k=2)
                new_out += [pred["sequence"] for pred in res]
            output = new_out
        return output


def gpt2_generation(input_text):
    tokenizer = AutoTokenizer.from_pretrained('macedonizer/sl-gpt2')
    model = AutoModelWithLMHead.from_pretrained('macedonizer/sl-gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    encoded_input = tokenizer(input_text, return_tensors="pt")
    output = model.generate(
        **encoded_input,
        bos_token_id=random.randint(1, 50000),
        do_sample=True,
        top_k=10,
        max_length=128,
        top_p=0.95,
        num_return_sequences=1,
    )

    decoded_output = []
    for sample in output:
        decoded_output.append(tokenizer.decode(sample, skip_special_tokens=True))

    for i in decoded_output: print(i)


def LSTM_generation(data, word2vec_path = "wiki.sl.vec"):
  model = load_keyed_vector_model(word2vec_path)
  generate_additional_data(data, "generated_token_lists.csv", model)
  data, text_classes = load("generated_token_lists.csv")
  tokenized_sent_list, Y = preprocess_tokens(data, generated=True)
  return tokenized_sent_list, Y
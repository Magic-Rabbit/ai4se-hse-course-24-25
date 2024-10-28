from pathlib import Path

import re
import datasets
import pandas as pd

contractions_dict = {"ain't": "is not", "aren't": "are not",
                       "can't": "cannot", "'cause": "because",
                       "could've": "could have", "couldn't": "could not",
                       "didn't": "did not", "doesn't": "does not",
                       "don't": "do not", "hadn't": "had not", "hasn't": "has not",
                       "haven't": "have not", "he'd": "he would", "he'll": "he will",
                       "he's": "he is", "how'd": "how did", "how'd'y": "how do you",
                       "how'll": "how will", "how's": "how is", "I'd": "I would",
                       "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
                       "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have",
                       "i'll": "i will", "i'll've": "i will have", "i'm": "i am",
                       "i've": "i have", "isn't": "is not", "it'd": "it would",
                       "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
                       "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not",
                       "might've": "might have", "mightn't": "might not",
                       "mightn't've": "might not have", "must've": "must have",
                       "mustn't": "must not", "mustn't've": "must not have",
                       "needn't": "need not", "needn't've": "need not have",
                       "o'clock": "of the clock", "oughtn't": "ought not",
                       "oughtn't've": "ought not have", "shan't": "shall not",
                       "sha'n't": "shall not", "shan't've": "shall not have",
                       "she'd": "she would", "she'd've": "she would have",
                       "she'll": "she will", "she'll've": "she will have",
                       "she's": "she is", "should've": "should have", "shouldn't": "should not",
                       "shouldn't've": "should not have", "so've": "so have", "so's": "so as",
                       "this's": "this is", "that'd": "that would", "that'd've": "that would have",
                       "that's": "that is", "there'd": "there would",
                       "there'd've": "there would have", "there's": "there is",
                       "here's": "here is", "they'd": "they would", "they'd've": "they would have",
                       "they'll": "they will", "they'll've": "they will have", "they're": "they are",
                       "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would",
                       "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                       "we're": "we are", "we've": "we have", "weren't": "were not",
                       "what'll": "what will",
                       "what'll've": "what will have", "what're": "what are", "what's": "what is",
                       "what've": "what have", "when's": "when is", "when've": "when have",
                       "where'd": "where did", "where's": "where is", "where've": "where have",
                       "who'll": "who will", "who'll've": "who will have", "who's": "who is",
                       "who've": "who have", "why's": "why is", "why've": "why have",
                       "will've": "will have", "won't": "will not", "won't've": "will not have",
                       "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
                       "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have",
                       "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would",
                       "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                       "you're": "you are", "you've": "you have", "aint": "is not", "arent": "are not",
                       "cant": "cannot", "cause": "because",
                       "couldve": "could have", "couldnt": "could not",
                       "didnt": "did not", "doesnt": "does not",
                       "dont": "do not", "hadnt": "had not", "hasnt": "has not",
                       "havent": "have not", "howdy": "how do you",
                       "its": "it is", "lets": "let us", "maam": "madam", "maynt": "may not",
                       "mightve": "might have", "mightnt": "might not",
                       "mightntve": "might not have", "mustve": "must have",
                       "mustnt": "must not", "mustntve": "must not have",
                       "neednt": "need not", "needntve": "need not have",
                       "oclock": "of the clock", "oughtnt": "ought not",
                       "shouldve": "should have", "shouldnt": "should not",
                       "werent": "were not", "yall": "you all", "youre": "you are",
                       "youve": "you have"}

profanity_dict = {
    r'(f)(u|[^a-z0-9 ])(c|[^a-z0-9 ])(k|[^a-z0-9 ])([^ ])*': 'fuck',
    r'(f)([^a-z]*)(u)([^a-z]*)(c)([^a-z]*)(k)': 'fuck',
    r' f[!@#\$%\^\&\*]*u[!@#\$%\^&\*]*k': 'fuck',
    r'f u u c': 'fuck',
    r'(f)(c|[^a-z ])(u|[^a-z ])(k)': 'fuck',
    r'f\*': 'fuck',
    r'feck ': 'fuck',
    r' fux ': 'fuck',
    r'f\*\*': 'fuck',
    r'f\-ing': 'fuck',
    r'f\.u\.': 'fuck',
    r'f###': 'fuck',
    r' fu ': 'fuck',
    r'f@ck': 'fuck',
    r'f u c k': 'fuck',
    r'f uck': 'fuck',
    r'f ck': 'fuck',
    r' (c)(r|[^a-z0-9 ])(a|[^a-z0-9 ])(p|[^a-z0-9 ])([^ ])*': 'crap',
    r' (c)([^a-z]*)(r)([^a-z]*)(a)([^a-z]*)(p)': 'crap',
    r' c[!@#\$%\^\&\*]*r[!@#\$%\^&\*]*p': 'crap',
    r'cr@p': 'crap',
    r' c r a p': 'crap',
    r'[^a-z]ass ': 'ass',
    r'[^a-z]azz ': 'ass',
    r'arrse': 'ass',
    r' arse ': 'ass',
    r'@\\$\\$': 'ass',
    r'[^a-z]anus': 'ass',
    r' a\*s\*s': 'ass',
    r'[^a-z]ass[^a-z ]': 'ass',
    r'a[@#\$%\^&\*][@#\$%\^&\*]': 'ass',
    r'[^a-z]anal ': 'ass',
    r'a s s': 'ass',
    r' a[s|z]*wipe': 'asshole',
    r'a[s|z]*[w]*h[o|0]+[l]*e': 'asshole',
    r'@\\$\\$hole': 'asshole',
    r'bitches': 'bitch',
    r' b[w]*i[t]*ch': 'bitch',
    r' b!tch': 'bitch',
    r' bi\+ch': 'bitch',
    r' b!\+ch': 'bitch',
    r' (b)([^a-z]*)(i)([^a-z]*)(t)([^a-z]*)(c)([^a-z]*)(h)': 'bitch',
    r' biatch': 'bitch',
    r' bi\*\*h': 'bitch',
    r' bytch': 'bitch',
    r'b i t c h': 'bitch',
    r'ba[s|z]+t[e|a]+rd': 'bastard',
    r'transgender': 'transgender',
    r'gay': 'gay',
    r'homo': 'gay',
    r'[^a-z]cock': 'cock',
    r'c0ck': 'cock',
    r'[^a-z]cok ': 'cock',
    r'c0k': 'cock',
    r'[^a-z]cok[^aeiou]': 'cock',
    r' cawk': 'cock',
    r'(c)([^a-z ])(o)([^a-z ]*)(c)([^a-z ]*)(k)': 'cock',
    r'c o c k': 'cock',
    r' dick[^aeiou]': 'dick',
    r'd i c k': 'dick',
    r'sucker': 'suck',
    r'(s)([^a-z ]*)(u)([^a-z ]*)(c)([^a-z ]*)(k)': 'suck',
    r'sucks': 'suck',
    r'5uck': 'suck',
    r's u c k': 'suck',
    r'cunt': 'cunt',
    r'c u n t': 'cunt',
    r'bullsh\*t': 'bullshit',
    r'bull\\$hit': 'bullshit',
    r'bull sh.t': 'bullshit',
    r'jerk': 'jerk',
    r'i[d]+io[t]+': 'idiot',
    r'(i)([^a-z ]*)(d)([^a-z ]*)(i)([^a-z ]*)(o)([^a-z ]*)(t)': 'idiot',
    r'idiots': 'idiot',
    r'i d i o t': 'idiot',
    r'(d)([^a-z ]*)(u)([^a-z ]*)(m)([^a-z ]*)(b)': 'dumb',
    r'shitty': 'shit',
    r'(s)([^a-z ]*)(h)([^a-z ]*)(i)([^a-z ]*)(t)': 'shit',
    r'shite': 'shit',
    r'\\$hit': 'shit',
    r's h i t': 'shit',
    r'sh\*tty': 'shit',
    r'sh\*ty': 'shit',
    r'sh\*t': 'shit',
    r'shythole': 'shit hole',
    r'sh.thole': 'shit hole',
    r'returd': 'retard',
    r'retad': 'retard',
    r'retard': 'retard',
    r'wiktard': 'retard',
    r'wikitud': 'retard',
    r'raped': 'rape',
    r'dumbass': 'dumb ass',
    r'dubass': 'dumb ass',
    r'butthead': 'ass head',
    r'sexy': 'sex',
    r's3x': 'sex',
    r'sexuality': 'sex',
    r'nigger': 'nigger',
    r'ni[g]+a': 'nigger',
    r' nigr ': 'nigger',
    r'negrito': 'nigger',
    r'niguh': 'nigger',
    r'n3gr': 'nigger',
    r'n i g g e r': 'nigger',
    r' stfu': 'shut the fuck up',
    r'^stfu': 'shut the fuck up',
    r' fyfi': 'for your fucking information',
    r'^fyfi': 'for your fucking information',
    r'gtfo': 'get the fuck off',
    r'^gtfo': 'get the fuck off',
    r' omfg': 'oh my fucking god',
    r'^omfg': 'oh my fucking god',
    r' wth': 'what the hell',
    r'^wth': 'what the hell',
    r' wtf': 'what the fuck',
    r'^wtf': 'what the fuck',
    r' sob ': 'son of bitch',
    r'^sob ': 'son of bitch',
    r'pussy[^c]': 'pussy',
    r'pusy': 'pussy',
    r'pussi[^l]': 'pussy',
    r'pusses': 'pussy',
    r'(p)(u|[^a-z0-9 ])(s|[^a-z0-9 ])(s|[^a-z0-9 ])(y)': 'pussy',
    r'faggot': 'faggot',
    r' fa[g]+[s]*[^a-z ]': 'faggot',
    r'fagot': 'faggot',
    r'f a g g o t': 'faggot',
    r'faggit': 'faggot',
    r'(f)([^a-z ]*)(a)([^a-z ]*)([g]+)([^a-z ]*)(o)([^a-z ]*)(t)': 'faggot',
    r'fau[g]+ot': 'faggot',
    r'fae[g]+ot': 'faggot',
    r' motha f': 'mother fucker',
    r' mother f': 'mother fucker',
    r'motherucker': 'mother fucker',
    r' mofo': 'mother fucker',
    r' mf ': 'mother fucker',
    r'wh\*\*\*': 'whore',
    r'w h o r e': 'whore',
    r'ha\*\*\*ha': 'haha',
}


def clean_text(text):
    # Удаление URL-ссылок
    text = re.sub(r'http\S+|www\S+', '', text)

    # Исправление сокращений
    for word, correction in contractions_dict.items():
        text = re.sub(r'\b' + word + r'\b', correction, text)

    # Удаление повторяющихся символов
    text = re.sub(r'(.)\1+', r'\1', text)

    # Удаление специальных символов
    text = re.sub(r'[&^#*]', '', text)

    # Исправление ругательных слов
    #for pattern, replacement in profanity_dict.items():
    #    text = re.sub(r'\b' + pattern + r'\b', replacement, text)

    return text

def prepare(raw_data: Path) -> datasets.Dataset:
    # Загрузка данных
    df = pd.read_excel(raw_data)

    # Удаление пропущенных значений и дубликатов
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # Очистка текста
    df['message'] = df['message'].apply(clean_text)

    # Преобразование DataFrame в Dataset
    dataset = datasets.Dataset.from_pandas(df)

    return dataset

def load_dataset(path: Path) -> datasets.Dataset:
    return datasets.load_from_disk(str(path))

def save_dataset(dataset: datasets.Dataset, path: Path) -> None:
    dataset.save_to_disk(str(path))


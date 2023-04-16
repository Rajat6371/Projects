# Tokenization of paragraphs/sentences
import nltk
import spacy
fr = spacy.load("fr_core_news_sm")

paragraph = """Les applications d'IA comprennent les moteurs de 
         recherche Web avancés (par exemple, Google), les 
         systèmes de recommandation (utilisés par YouTube, 
         Amazon et Netflix), la compréhension de la parole 
         humaine (comme Siri et Alexa), les voitures autonomes 
         (par exemple, Waymo), la prise de décision automatisée
         et la compétition au plus haut niveau dans les systèmes
         de jeux stratégiques (tels que les échecs et Go).[2] À
         mesure que les machines deviennent de plus en plus 
         performantes, les tâches considérées comme nécessitant
         « l'intelligence » sont souvent retirées de la définition 
         de l'IA, un phénomène connu sous le nom d'effet IA.
         [3] Par exemple, la reconnaissance optique de caractères 
         est souvent exclue des choses considérées comme de 
         l'IA,[4] étant devenue une technologie de routine.[5] """
               
# Tokenizing sentences
sentences = nltk.sent_tokenize(paragraph)

# Tokenizing words
words = nltk.word_tokenize(paragraph)

# 🤖 Personal AI Assistant

Καλώς ήρθατε στον Personal AI Assistant, έναν έξυπνο βοηθό που έχει δημιουργηθεί για να απλοποιεί τις καθημερινές σας εργασίες! Αυτή η εφαρμογή χρησιμοποιεί το LangChain και το DeepSeek AI μοντέλο (μέσω OpenRouter) για να παρέχει ένα ευρύ φάσμα λειτουργιών, από τη διαχείριση του ημερολογίου σας έως την απάντηση σε ερωτήσεις βασισμένες σε έγγραφα.

## ✨ Δυνατότητες

Ο Personal AI Assistant μπορεί να εκτελέσει τις ακόλουθες εργασίες:

- **Διαχείριση Ημερολογίου:** Προγραμματίστε συναντήσεις και εκδηλώσεις
- **Υπενθυμίσεις:** Ορίστε εξατομικευμένες υπενθυμίσεις για σημαντικές εργασίες
- **Σύνοψη Κειμένου:** Συνοψίστε γρήγορα μεγάλα κείμενα
- **Επανεγγραφή Κειμένου:** Αλλάξτε το ύφος ή τον τόνο ενός κειμένου (π.χ., επίσημο, φιλικό, για social media)
- **Πρόγνωση Καιρού:** Λάβετε ενημερώσεις καιρού για οποιαδήποτε πόλη
- **Απάντηση από Έγγραφα (RAG):** Φορτώστε αρχεία PDF και υποβάλετε ερωτήσεις για το περιεχόμενό τους
- **Σύνταξη Email:** Ζητήστε από τον βοηθό να συντάξει ένα προσχέδιο email

## 🚀 Γρήγορη Εκκίνηση

Για να ξεκινήσετε τον Personal AI Assistant, χρησιμοποιήστε το Docker.

### Προαπαιτούμενα

- **Docker** εγκατεστημένο στο σύστημά σας
- Ένα **API Key** από το OpenRouter. Μπορείτε να βρείτε το κλειδί σας στο dashboard τους
- Δημιουργήστε ένα αρχείο `.env` στο root directory του project σας με το OpenRouter API Key:

```env
OPENROUTER_API_KEY="sk-or-v1-YOUR_OPENROUTER_API_KEY_HERE"
```

**Σημαντικό:** Προσθέστε αυτό το αρχείο στο `.gitignore` για να μην ανέβει ποτέ στο Git repository σας.

### Βήματα Εκτέλεσης

1. **Κλωνοποιήστε το Repository:**
   ```bash
   git clone <URL_του_repository_σας>
   cd <όνομα_φακέλου_project>
   ```

2. **Δημιουργήστε το Docker Image:**
   ```bash
   docker build -t ai-personal-assistant .
   ```

3. **Εκτελέστε τον Docker Container:**
   ```bash
   docker run -it --rm -p 8888:8888 -e OPENROUTER_API_KEY="sk-or-v1-YOUR_OPENROUTER_API_KEY_HERE" -v "%cd%":/app ai-personal-assistant
   ```
   
   - `-e OPENROUTER_API_KEY="..."`: Αυτό το flag περνάει το API key ως environment variable στον container
   - `-v "%cd%":/app`: Αυτό το volume mount επιτρέπει στον container να έχει πρόσβαση στα αρχεία του τρέχοντος καταλόγου (π.χ., τα PDF που θέλετε να φορτώσετε για RAG)

4. **Πρόσβαση στον Βοηθό:**
   Αφού ο container εκτελεστεί, ανοίξτε το πρόγραμμα περιήγησής σας και μεταβείτε στη διεύθυνση:
   ```
   http://localhost:8888
   ```
   
   Θα δείτε τη διεπαφή συνομιλίας του Gradio έτοιμη για χρήση!

## 💡 Παραδείγματα Χρήσης

Δοκιμάστε τις ακόλουθες εντολές στο chat interface:

- `"Schedule a meeting tomorrow at 10 AM, 'Project Sync'"`
- `"Set a reminder for Tuesday at 3 PM: 'Call John'"`
- `"Write an email to support@example.com with subject 'Issue Report' and body 'I am experiencing a login problem.'"`
- `"Summarize the following text: Artificial intelligence is changing the world."`
- `"Tell me the weather for Athens today."`
- `"load file: my_document.pdf"` (Αντικαταστήστε το `my_document.pdf` με το πραγματικό όνομα του αρχείου PDF που έχετε στον φάκελο του project σας)
- `"Answer from the document: Who built the Eiffel Tower?"` (Αφού έχετε φορτώσει ένα έγγραφο)

## 🛠️ Τεχνολογίες που Χρησιμοποιούνται

- **Python:** Η βασική γλώσσα προγραμματισμού
- **LangChain:** Ένα πλαίσιο για την ανάπτυξη εφαρμογών με μοντέλα γλώσσας
- **OpenRouter:** Παρέχει πρόσβαση σε διάφορα LLM μοντέλα, συμπεριλαμβανομένου του DeepSeek-Chat
- **DeepSeek-Chat-v3-0324:** Το μεγάλο γλωσσικό μοντέλο που χρησιμοποιείται για την επεξεργασία και παραγωγή κειμένου
- **SentenceTransformerEmbeddings:** Για τη δημιουργία διανυσματικών αναπαραστάσεων κειμένου (embeddings)
- **ChromaDB:** Ένα vector store για την αποθήκευση και αναζήτηση ενσωματώσεων εγγράφων
- **PyPDFLoader:** Για τη φόρτωση εγγράφων PDF
- **Gradio:** Για τη δημιουργία μιας εύχρηστης web διεπαφής χρήστη
- **Docker:** Για την containerization της εφαρμογής

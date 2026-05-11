# Skrypt prezentacji — AntyDezinformator

**Autor:** Vadym Abrosimov | **Czas:** ~6 minut

---

## Slajd 1 — Tytuł

Dzień dobry. Nazywam się Vadym Abrosimov i chciałbym przedstawić projekt inżynierski zatytułowany „System detekcji dezinformacji z wyjaśnialną sztuczną inteligencją", który realizowałem pod kierunkiem dr. inż. Andrzeja Burdy.

---

## Slajd 2 — Cel projektu

Projekt realizuje trzy powiązane cele.

Po pierwsze — automatyczna **detekcja technik manipulacji**: system identyfikuje jedenaście kategorii dezinformacji w polskojęzycznych tekstach, takich jak propaganda emocjonalna, selektywny dobór danych czy manipulacja źródłami.

Po drugie — **wyjaśnialne uzasadnienia**: w odróżnieniu od klasycznych klasyfikatorów, system nie zwraca tylko etykiety, lecz generuje czytelne uzasadnienie decyzji w języku naturalnym — tzw. Chain-of-Thought.

Po trzecie — **kompletna pętla MLOps**: system umożliwia douczanie modelu przez eksperta w trybie Human-in-the-Loop, z automatyczną ewaluacją i kontrolowanym wdrożeniem.

---

## Slajd 3 — Motywacja

Dlaczego to potrzebne? Istniejące rozwiązania mają poważne luki.

Modele klasyfikacyjne oparte na architekturze BERT, jak PL-RoBERTa, zwracają wyłącznie wynik liczbowy — bez żadnego wyjaśnienia. Użytkownik dostaje etykietę „manipulacja", ale nie wie dlaczego.

Modele chmurowe, takie jak GPT-4, posiadają restrykcyjne filtry, które często odmawiają analizy kontrowersyjnych treści politycznych — a właśnie takie są głównym nośnikiem dezinformacji. Do tego dochodzą koszty API i ryzyko prywatności.

Ręczny fact-checking to „złoty standard" wiarygodności, ale weryfikacja jednego artykułu zajmuje godziny — jest całkowicie nieskalowalna.

**Żadne z tych rozwiązań nie łączy jednocześnie**: lokalnego wdrożenia, wyjaśnialności i zdolności adaptacji do nowych danych. Tę lukę wypełnia mój system.

---

## Slajd 4 — Stos technologiczny

Wybór technologii był podyktowany dwoma ograniczeniami: sprzęt konsumencki i język polski.

Jako model produkcyjny wybrałem **Bielik-4.5B** — model specjalnie dostrojony do języka polskiego, zdolny do działania na karcie graficznej klasy konsumenckiej. Do generowania danych treningowych użyłem większego modelu **Qwen-2.5-7B** jako nauczyciela.

Trenowanie przeprowadziłem techniką **QLoRA z biblioteką Unsloth** na darmowej instancji Google Colab — co czyni projekt powtarzalnym bez dostępu do drogiej infrastruktury.

Inferencja działa lokalnie przez **Ollama** z formatem GGUF — bez połączenia z chmurą, z pełną gwarancją prywatności danych.

Warstwę aplikacyjną tworzy **FastAPI** po stronie backendu i **React** na frontendzie, z bazą SQLite do śledzenia historii treningów.

---

## Slajd 5 — Architektura

Architektura systemu składa się z dwóch przepływów.

**Główny przepływ inferencji**: użytkownik wkleja tekst artykułu w interfejsie React, żądanie trafia do backendu FastAPI, który wywołuje model Bielik z adapterem LoRA przez Ollama. Model zwraca strukturę JSON z wykrytymi technikami i uzasadnieniem, a backend normalizuje odpowiedź przed wyświetleniem.

**Pętla MLOps** to unikalny element systemu: ekspert wgrywa nowe dane treningowe, co uruchamia douczanie — które działa w środowisku WSL2 — po zakończeniu automatycznie odpala się benchmark na zbiorze testowym, a wyniki F1 trafiają z powrotem do interfejsu. Ekspert sam decyduje o wdrożeniu nowego adaptera poprzez mechanizm hot-swap w serwerze Ollama.

---

## Slajd 6 — Niezawodność

Jednym z kluczowych wyzwań inżynierskich była niestabilność wyjścia modelu językowego. Model generuje tekst, nie strukturę danych — więc wynik może zawierać błędy składniowe JSON lub literówki w nazwach etykiet.

Zaprojektowałem trójfazowy potok naprawczy: najpierw ścisłe parsowanie JSON, potem odzyskiwanie przez wyrażenia regularne przy drobnych błędach strukturalnych, a na końcu normalizacja nazw tagów — na przykład automatyczna korekta `EMOTIORAL` na `EMOTIONAL`.

Efektem jest **wskaźnik poprawnej struktury powyżej 96%** — czyli niemal każda odpowiedź modelu jest prawidłowo odczytana i zwrócona użytkownikowi.

---

## Slajd 7 — Wyniki

Przeprowadziłem ewaluację na zbiorze testowym liczącym **1521 próbek** z korpusu MIPD, porównując trzy warianty modelu.

Model bazowy bez adaptera praktycznie nie nadaje się do zastosowań produkcyjnych — wskaźnik PSR wynosi zaledwie 20%, a F1 — 0,10.

**Prototype Adapter**, dostrojony wyłącznie pod kątem klasyfikacji, osiąga F1 równe **0,49** — co przewyższa wyniki PL-RoBERTa-Large raportowane w literaturze dla tego samego zbioru.

**XAI Adapter** — wersja z warstwą wyjaśnialności — osiąga niższe F1 równe 0,28, przy zachowaniu dokładności strukturalnej na poziomie 73%. Ten spadek to tzw. **podatek od wyjaśnialności**: dodanie uzasadnień zwiększa złożoność zadania dla modelu, a jakość danych syntetycznych wygenerowanych przez nauczyciela okazała się kluczowym czynnikiem ograniczającym.

---

## Slajd 8 — Podsumowanie

Podsumowując: projekt dostarczył **kompletny, działający system inżynierski** — od inferencji, przez douczanie, po automatyczną ewaluację i hot-swap modelu na produkcji.

Kluczowe właściwości to pełna lokalność, prywatność danych i zdolność adaptacji do nowych narracji dezinformacyjnych.

Główne kierunki dalszego rozwoju to migracja na sprzęt serwerowy z większą ilością VRAM, wdrożenie metodologii LLM-as-a-Judge do automatycznej oceny jakości uzasadnień, rozbudowa mechanizmów bezpieczeństwa o kontrolę dostępu opartą na rolach oraz regeneracja syntetycznego zbioru danych uzasadnień przy użyciu modelu flagowego (np. GPT-4) zamiast lokalnego modelu 7B — co pozwoliłoby podnieść jakość i różnorodność reasoningu trenującego.

Dziękuję za uwagę. Chętnie odpowiem na pytania.

---

_Łączny czas: ~6 minut przy tempie ~130 słów/minutę_

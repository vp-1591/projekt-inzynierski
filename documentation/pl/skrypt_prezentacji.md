# Skrypt prezentacji — System detekcji manipulacji

**Autor:** Vadym Abrosimov | **Czas:** ~6 minut

---

## Slajd 1 — Tytuł

Dzień dobry. Nazywam się Vadym Abrosimov i chciałbym przedstawić projekt inżynierski „System detekcji manipulacji”.

Pełny temat pracy to „System detekcji dezinformacji z wyjaśnialną sztuczną inteligencją”, natomiast w prezentacji używam krótszego określenia „System detekcji manipulacji”, ponieważ system operacyjnie wykrywa konkretne techniki manipulacyjne występujące w treściach dezinformacyjnych.

Celem pracy było stworzenie lokalnego systemu wykrywania manipulacji medialnych, który nie tylko klasyfikuje tekst, ale także uzasadnia decyzję i może być iteracyjnie douczany.

---

## Slajd 2 — Motywacja

Problem dezinformacji polega nie tylko na tym, że treści manipulacyjne są trudne do wykrycia, ale również na tym, że sama etykieta „manipulacja” często nie wystarcza.

Klasyczne modele klasyfikacyjne, takie jak rozwiązania oparte na BERT, zwykle zwracają wynik liczbowy albo etykietę, ale nie wyjaśniają, dlaczego dana decyzja została podjęta.

Modele chmurowe, na przykład GPT-4, mają z kolei ograniczenia moderacji treści i mechanizmy bezpieczeństwa, a dodatkowo wiążą się z kosztami API oraz ryzykiem przekazywania danych poza lokalne środowisko.

Ręczny fact-checking jest bardzo dokładny, ale trudno go skalować, ponieważ analiza jednego artykułu może wymagać wielu godzin pracy eksperta.

Dlatego w pracy skupiłem się na luce pomiędzy tymi podejściami: lokalnym systemie, który wykrywa techniki manipulacji, pokazuje uzasadnienie i może być dostosowywany do nowych przykładów.

---

## Slajd 3 — Cel projektu

Projekt realizuje trzy główne cele.

Po pierwsze, system ma wykrywać techniki manipulacji w polskojęzycznych tekstach. Chodzi nie tylko o stwierdzenie, czy tekst jest problematyczny, ale o wskazanie konkretnego typu manipulacji.

Po drugie, system ma generować wyjaśnialne uzasadnienia. Użytkownik powinien zobaczyć, co w tekście wpłynęło na decyzję modelu, a nie tylko końcową etykietę.

Po trzecie, system ma umożliwiać douczanie. Ekspert może dodać nowe przykłady, a model może zostać ponownie dostrojony, gdy pojawiają się nowe typy narracji lub manipulacji.

---

## Slajd 4 — Architektura

Architektura systemu składa się z dwóch trybów: analizy tekstu oraz douczania eksperckiego.

Pierwszy tryb to normalna praca systemu. Użytkownik wkleja tekst artykułu w aplikacji, a system przekazuje go do lokalnego modelu językowego. Model analizuje treść i zwraca wynik w dwóch częściach: wykryte techniki manipulacji oraz krótkie uzasadnienie decyzji.

Po stronie aplikacji wynik nie jest pokazywany bezpośrednio w surowej postaci. Backend sprawdza, czy odpowiedź ma poprawną strukturę, porządkuje etykiety i dopiero wtedy przekazuje wynik do interfejsu użytkownika.

Drugi tryb dotyczy rozwoju systemu. Ekspert może dostarczyć nowe, opisane przykłady manipulacji, a następnie uruchomić douczanie modelu. Po zakończeniu system wykonuje ewaluację, czyli sprawdza, czy nowa wersja rzeczywiście działa lepiej.

Wdrożenie nowej wersji nie jest automatyczne. Ekspert widzi wynik oceny i dopiero na tej podstawie decyduje, czy nowy adapter ma zastąpić poprzednią wersję.

Najważniejsze jest to, że oba przepływy działają lokalnie i nie wymagają wysyłania analizowanych treści do zewnętrznych usług.

---

## Slajd 5 — Przykład działania systemu

Na tym slajdzie pokazuję prosty przykład działania aplikacji.

Na wejściu system otrzymuje fragment tekstu artykułu. Następnie model analizuje treść i wskazuje wykrytą technikę manipulacji — w tym przykładzie jest to wybiórczość, czyli cherry picking.

System nie kończy jednak na samej etykiecie. Pod spodem generuje krótkie uzasadnienie, które wyjaśnia, dlaczego dany fragment został zaklasyfikowany w ten sposób.

To jest istotna różnica względem typowego klasyfikatora: użytkownik może ocenić nie tylko wynik, ale również logikę stojącą za decyzją modelu.

---

## Slajd 6 — Stos technologiczny

Najważniejszym elementem jest lokalny model językowy Bielik, wybrany ze względu na obsługę języka polskiego i możliwość uruchomienia na sprzęcie konsumenckim.

Douczanie zrealizowałem w sposób oszczędny obliczeniowo: zamiast trenować cały model od zera, system uczy niewielki adapter. Dzięki temu eksperymenty były możliwe bez drogiej infrastruktury serwerowej.

Do uruchamiania modelu lokalnie wykorzystałem Ollama i llama.cpp. Sama aplikacja składa się z części serwerowej, interfejsu użytkownika oraz prostej bazy danych do zapisywania historii treningów i eksperymentów.

Najważniejsze z perspektywy pracy nie są jednak same nazwy bibliotek, ale efekt: cały przepływ analizy i douczania może działać lokalnie.

---

## Slajd 7 — Niezawodność

Jednym z praktycznych problemów było to, że model językowy generuje tekst, a aplikacja potrzebuje uporządkowanych danych.

Model może więc odpowiedzieć sensownie, ale w formacie, którego aplikacja nie potrafiłaby od razu odczytać.

Dlatego dodałem warstwę sprawdzającą odpowiedź modelu. Jeśli format jest niepoprawny, system próbuje go automatycznie naprawić i ujednolicić nazwy etykiet.

Efektem jest wskaźnik poprawnej struktury powyżej 96%. Ten slajd pokazuje więc nie tyle detal implementacyjny, ile mechanizm, który pozwala używać modelu w stabilnej aplikacji.

---

## Slajd 8 — Wyniki

Ewaluację przeprowadziłem na zbiorze testowym liczącym 1521 próbek.

Model bazowy bez dostrojenia osiągał słabe wyniki. To pokazuje, że samo użycie modelu językowego nie wystarcza do tego zadania.

Po douczeniu wynik wzrósł do F1 równego 0,49, przy wysokiej poprawności struktury odpowiedzi.

Wynik należy interpretować ostrożnie, ponieważ porównania z literaturą nie zawsze są w pełni równoważne metodologicznie. Można jednak powiedzieć, że uzyskane rezultaty są konkurencyjne wobec raportowanych wyników, zwłaszcza biorąc pod uwagę lokalny model 4,5B.

Wariant generujący uzasadnienia osiągnął niższy wynik. Jest to koszt wyjaśnialności: model wykonuje trudniejsze zadanie, bo musi jednocześnie wskazać technikę i wyjaśnić swoją decyzję.

---

## Slajd 9 — Podsumowanie

Podsumowując, w ramach pracy powstał kompletny system: od lokalnej inferencji, przez douczanie, po ewaluację i kontrolowane wdrożenie adaptera.

Najważniejsze cechy projektu to lokalne działanie, prywatność danych, wyjaśnialność decyzji oraz możliwość adaptacji systemu do nowych przykładów.

Główne kierunki dalszego rozwoju to użycie większych modeli na mocniejszym GPU, automatyczna ocena jakości uzasadnień oraz rozbudowa mechanizmów bezpieczeństwa aplikacji.

Projekt potwierdził możliwość implementacji lokalnego systemu XAI opartego na LLM z pełnym cyklem MLOps na sprzęcie klasy konsumenckiej.

Dziękuję za uwagę. Chętnie odpowiem na pytania.

---

_Łączny czas: ~6 minut przy tempie około 130 słów/minutę_

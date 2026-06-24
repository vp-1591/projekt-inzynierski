# Skrypt prezentacji — System detekcji manipulacji

**Autor:** Vadym Abrosimov | **Czas:** ~6 minut

---

## Slajd 1 — Tytuł

Dzień dobry. Nazywam się Vadym Abrosimov i chciałbym przedstawić projekt inżynierski „System detekcji manipulacji”.

Celem pracy było stworzenie lokalnego systemu wykrywania manipulacji medialnych, który nie tylko ocenia tekst, ale także wyjaśnia swoją decyzję i może być rozwijany na nowych przykładach.

---

## Slajd 2 — Motywacja

Problem dezinformacji polega nie tylko na tym, że treści manipulacyjne są trudne do wykrycia, ale również na tym, że sama etykieta „manipulacja” często nie wystarcza.

Typowe narzędzia do oceniania tekstu zwykle zwracają wynik liczbowy albo krótką etykietę, ale nie wyjaśniają, dlaczego podjęły taką decyzję.

Rozwiązania działające w chmurze, na przykład GPT-4, mają z kolei własne ograniczenia bezpieczeństwa, generują koszty użycia i wymagają wysyłania danych poza komputer użytkownika.

Ręczne sprawdzanie faktów jest bardzo dokładne, ale trudno robić je na dużą skalę, ponieważ analiza jednego artykułu może wymagać wielu godzin pracy eksperta.

Dlatego w pracy skupiłem się na rozwiązaniu pośrednim: lokalnym systemie, który wykrywa techniki manipulacji, pokazuje uzasadnienie i może być dostosowywany do nowych przykładów.

---

## Slajd 3 — Cel projektu

Projekt realizuje trzy główne cele.

Po pierwsze, system ma wykrywać techniki manipulacji w polskojęzycznych tekstach. Chodzi nie tylko o stwierdzenie, czy tekst jest problematyczny, ale o wskazanie, na czym polega problem.

Po drugie, system ma podawać zrozumiałe uzasadnienia. Użytkownik powinien zobaczyć, co w tekście wpłynęło na decyzję, a nie tylko końcowy wynik.

Po trzecie, system ma umożliwiać adaptacje. Ekspert może dodać nowe przykłady, a system może zostać zaktualizowany, gdy pojawiają się nowe sposoby manipulacji.

---

## Slajd 4 — Architektura

System działa w dwóch trybach: analizy tekstu oraz uczenia na przykładach przygotowanych przez eksperta.

Pierwszy tryb to normalna praca systemu. Użytkownik wkleja tekst artykułu w aplikacji, a system analizuje go lokalnie na komputerze. Wynik składa się z dwóch części: wykrytych technik manipulacji oraz krótkiego uzasadnienia decyzji.

Po stronie aplikacji wynik nie jest pokazywany od razu w surowej postaci. Część serwerowa sprawdza, czy odpowiedź jest poprawnie zapisana, porządkuje nazwy wykrytych technik i dopiero wtedy pokazuje wynik użytkownikowi.

Drugi tryb dotyczy rozwoju systemu. Ekspert może dostarczyć nowe, opisane przykłady manipulacji, a następnie uruchomić dodatkowe uczenie. Po zakończeniu system sprawdza, czy nowa wersja rzeczywiście działa lepiej.

Wdrożenie nowej wersji nie jest automatyczne. Ekspert widzi wynik oceny i dopiero na tej podstawie decyduje, czy zastąpić poprzednią wersję.

Najważniejsze jest to, że oba przepływy działają lokalnie i nie wymagają wysyłania analizowanych treści do zewnętrznych usług.

---

## Slajd 5 — Przykład działania systemu

Na tym slajdzie pokazuję prosty przykład działania aplikacji.

Na wejściu system otrzymuje fragment tekstu artykułu. Następnie analizuje treść i wskazuje wykrytą technikę manipulacji. W tym przykładzie jest to wybiórczość, czyli pokazywanie tylko tych faktów, które pasują do danej tezy.

System nie kończy jednak na samej nazwie techniki. Pod spodem pokazuje krótkie uzasadnienie, które wyjaśnia, dlaczego dany fragment został oceniony w ten sposób.

To jest istotna różnica względem prostego narzędzia, które zwraca tylko wynik: użytkownik może ocenić nie tylko decyzję, ale również jej uzasadnienie.

---

## Slajd 6 — Stos technologiczny

Najważniejszym elementem jest lokalny model językowy Bielik, wybrany ze względu na dobrą obsługę języka polskiego i możliwość uruchomienia na zwykłym komputerze z odpowiednią kartą graficzną.

Dalsze uczenie zrealizowałem w sposób oszczędny: zamiast uczyć cały model od zera, system uczy tylko niewielki dodatkowy element. Dzięki temu eksperymenty były możliwe bez drogiej infrastruktury serwerowej.

Do uruchamiania modelu lokalnie wykorzystałem Ollama i llama.cpp. Sama aplikacja składa się z części serwerowej, widoku dla użytkownika oraz prostej bazy danych do zapisywania historii uczenia i eksperymentów.

---

## Slajd 7 — Niezawodność

Jednym z praktycznych problemów było to, że model językowy odpowiada tekstem, a aplikacja potrzebuje danych zapisanych w przewidywalny sposób. Model może więc odpowiedzieć sensownie, ale w formie, której aplikacja nie potrafiłaby od razu odczytać.

Dlatego dodałem warstwę sprawdzającą i naprawiaca odpowiedź modelu. Efektem jest poprawny zapis odpowiedzi w ponad 96% przypadków. 

---

## Slajd 8 — Wyniki

Skuteczność sprawdziłem na zbiorze testowym liczącym 1521 próbek.

Model bazowy, czyli wersja bez dodatkowego uczenia, osiągał słabe wyniki. Po dodaniu przykładów skuteczność wzrosła do F1 równego 0,49. Jest to miara, która łączy trafność wykrywania z liczbą pomyłek.

Ten wynik należy porównywać ostrożnie, bo inne prace często używają trochę innych danych i zasad oceny. Można jednak powiedzieć, że rezultat jest konkurencyjny, zwłaszcza jak na lokalny model tej wielkości.

Wariant generujący uzasadnienia osiągnął niższy wynik. To pokazuje koszt wyjaśniania decyzji: system wykonuje trudniejsze zadanie, bo musi jednocześnie wskazać technikę i wyjaśnić, dlaczego ją wybrał. Dodatknie, potencjalnie nieprecyzyjne rozumowanie (CoT) w danych syntetycznych mogło wpłynąć na jakość etykiet, co przekłada się na metryki modelu adaptera.

---

## Slajd 9 — Podsumowanie

Podsumowując, w ramach pracy powstał działający system, który lokalnie analizuje polskie teksty, wskazuje możliwe manipulacje i pokazuje powód swojej decyzji.

Najważniejsze jest dla mnie to, że użytkownik nie dostaje tylko odpowiedzi „tak” albo „nie”. Dostaje też krótkie wyjaśnienie, a ekspert może później rozwijać system na nowych przykładach.

W dalszej pracy skupiłbym się na mocniejszym GPU do trenowania adapterow do większych modeli, mechanizmie LLM-as-a-judge do weryfikacji spójności CoT z etykietami, rozdzieleniu interfejsu eksperta od użytkownika oraz konteneryzacji z Dockerem dla lepszej przenoszalności.

Dziękuję za uwagę. Chętnie odpowiem na pytania.

---

# Opis działania programu modelów klasyfikujących kardiomegalię 






## Przetwarzanie danych

Przed rozpoczęciem programowania modelów dane zostały podzielone w proporcji 80:20 odpowiednio na treningowe i testowe oraz zostały przeskalowane przy użyciu skalera standardowego.


## Decision Tree

Klasyfikator Decision Tree dzieli dane poprzez stawianie im warunków aż do momentu czystego ich podziału. Decision Tree zaczyna się od tzw. korzenia, czyli pierwszego węzła stawiającego warunek, a kończy na liściach, które reprezentują wszystkie możliwe rezultaty jakie można uzyskać z danej bazy danych. Aby osiągnąć najlepsze wyniki, zostały użyte funkcje RepeatedStratifiedKFold() oraz GridSearchCV().


## K-Nearest Neighbour (KNN)

Klasyfikator KNN ocenia dane na zasadzie ich odległości od innych danych. KNN klasyfikuje więc daną badając jej otoczenie i dopasowując ją do do najliczniejszej grupy danych znajdującej się w badanej okolicy. Do tuningu modelu zostały wykorzystane funkcje RepeatedStratifiedKFold() oraz GridSearchCV().


## Support Vector Classification (SVC)

Klasyfikator SVC tworzy hiperpłaszczyznę rozdzielającą dane na dwie grupy. Optymalna hiperpłaszcznyzna maksymalizuje margines, czyli odległość hiperpłaszczyzny od najbliższego wektora danych. Aby znaleźć odpowiednie hiperparametry do tego modelu zostały zastosowane funkcje RepeatedStratifiedKFold() oraz GridSearchCV().


## Sposób wykorzystania każdego z modelów

Aby znaleźć odpowiednie hiperparametry dla każdego modelu zostały zastosowane funkcje RepeatedStratifiedKFold() oraz GridSearchCV(), które testują różne kombinacje parametrów i zwracają najlepsze wyniki. Modele były trenowane na treningowej części danych, a finalnie sprawdzane na testowej. Wszystko jest sprawdzone przy użyciu kroswalidacji (sprawdzania krzyżowego)


## Finalne wyniki

Program zwraca informację o średniej wartości kroswalidacji i odchylenia standardowego każdego modelu na danych treningowych oraz wynik dokładności każdego modelu na danych testowych.

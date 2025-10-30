# Opis działania programu modeli klasyfikujących kardiomegalię 






## Przetwarzanie danych

Przed rozpoczęciem programowania modeli dane zostały podzielone w proporcji 80:20 odpowiednio na treningowe i testowe oraz zostały przeskalowane przy użyciu skalera standardowego.


## Decision Tree

Klasyfikator Decision Tree dzieli dane poprzez stawianie im warunków aż do momentu czystego ich podziału. Decision Tree zaczyna się od tzw. korzenia, czyli pierwszego węzła stawiającego warunek, a kończy na liściach, które reprezentują wszystkie możliwe rezultaty jakie można uzyskać z danej bazy danych.


## K-Nearest Neighbour (KNN)

Klasyfikator KNN ocenia dane na zasadzie ich odległości od innych danych. KNN klasyfikuje więc daną badając jej otoczenie i dopasowując ją do do najliczniejszej grupy danych znajdującej się w badanej okolicy. 

## Support Vector Classification (SVC)

Klasyfikator SVC tworzy hiperpłaszczyznę rozdzielającą dane na dwie grupy. Optymalna hiperpłaszcznyzna maksymalizuje margines, czyli odległość hiperpłaszczyzny od najbliższego wektora danych. 


## Sposób wykorzystania każdego z modelów

Aby znaleźć odpowiednie hiperparametry dla każdego modelu zostały zastosowane funkcje RepeatedStratifiedKFold() oraz GridSearchCV(), które testują różne kombinacje parametrów i zwracają najlepsze wyniki. Modele były trenowane na treningowej części danych, a finalnie sprawdzane na testowej. Wszystko jest sprawdzone przy użyciu kroswalidacji (sprawdzania krzyżowego), która polega na podziale danych na mniejsze grupy w celu przeprowadzeniu testów.


## Finalne wyniki

Program zwraca informację o średniej wartości kroswalidacji i odchylenia standardowego każdego modelu na danych treningowych oraz wynik dokładności każdego modelu na danych testowych. 

Wyniki modeli:

<img width="225" height="188" alt="image" src="https://github.com/user-attachments/assets/a0b854fa-ada9-4ae4-9b6b-184e49f75154" />\

Jak widać powyżej, najlepszy wynik dokładności uzyskał model stosujący SVC. Na wyniki, poza sposobami pracy danych modeli i hiperparametrami, wpływ miała również m.in. ilość danych.




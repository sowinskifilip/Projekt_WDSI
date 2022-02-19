# Projekt_WDSI
**Autor: Filip Sowinski**

**Założony cel projketu:**
Wykonany projekt realizuje zadanie klasyfikacji znaków na dwie grupy: znaki ograniczenia prędkości oraz pozostałe.

**Sposób obsługi projektu:**
Po uruchuomieniu pliku main.py należy w konsoli wpisać ciąg znaków 'classify', który pozwoli na wczytanie danych testowych zgodnie z przyjętą strukturą przykładowego wejścia w instrukcji. Po zakończeniu etapu predykcji dane zwracane są poprzez konsolę zgodnie z przykładowym formatem wyjścia.

**Opis działania algorytmu:**
Działanie algorytmu opiera się na początkowej fazie odpowiedzialnej za przetworzenie danych treningowych w taki sposób, by charakterystyczne punkty zamieszczone w plikach xml zostały wycięte, a następnie zapisane w przyjętej strukturze danych jako wycinek obrazu, gdzie zostają oetykietowane w zależności od tego, czy prezentują obiekt typu 'speedlimit' w pliku xml czy też nie. Niestety nie udało się zrealizować dodawania losowych wycinków ze zdjęć. Uzyskane w opisany sposób dane podlegają procesowi ekstrakcji cech oraz klasteryzacji. Implementacyjnie zostało to zrealizowane przy użyciu gotowych algorytmów z biblioteki OpenCV, tj: BOWKMeansTrainer, BOWImgDescriptorExtractor, SIFT, FlannBasedMatcher. Ostatecznie przetworzone dane opisane są poprzez deskryptory. Kolejnym etapem jest nauczenie modelu w postaci drzewa decyzyjnego zaimplementowanego już w bibliotece sklearn: RandomForestClassifier na podstawie przetworzonych danych. Finalnym etapem działania programu jest wczytanie z poziomu konsoli danych testowych, które podleją procesowi ekstrakcji cech, a następnie predykcji zrealizowanej przez nauczony model, który zwraca wynik klasyfikacji z poziomu konsoli.

 

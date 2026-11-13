# Znaki wodne — pięć metod

## 1. [A Novel Improved Reversible Visible Image Watermarking Algorithm Based on Grad-CAM and JND](https://doi.org/10.1155/2021/6652897)

- **Problem:** Widoczne oznaczanie obrazów przy zachowaniu jakości i możliwości bezstratnego odzyskania oryginału.
- **Algorytm:** Grad-CAM wskazuje istotny obszar obrazu, a model JND dostosowuje intensywność znaku do percepcji wzrokowej. Odwracalne ukrywanie danych przechowuje informacje potrzebne do odtworzenia obrazu.
- **Co ciekawe:** Łączy widoczny znak z analizą zawartości obrazu i odwracalnością modyfikacji.

## 2. [StegaStamp: Invisible Hyperlinks in Physical Photographs](https://arxiv.org/abs/1904.05343)

- **Problem:** Ukrywanie identyfikatorów odsyłających do stron internetowych w zdjęciach, także po wydrukowaniu i ponownym sfotografowaniu.
- **Algorytm:** Enkoder neuronowy osadza bity w obrazie, a dekoder je odczytuje. Trening symuluje zniekształcenia fotografowania; korekcja błędów pomaga odzyskać wiadomość.
- **Co ciekawe:** Niemal niewidoczny kod działa również w świecie fizycznym, bez widocznego kodu QR.

## 3. [Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust](https://arxiv.org/abs/2305.20030)

- **Problem:** Rozpoznawanie obrazów wygenerowanych przez oznaczony model dyfuzyjny, również po przekształceniach obrazu.
- **Algorytm:** W początkowym szumie osadza się wzór pierścieni w dziedzinie Fouriera. Detektor odwraca proces dyfuzji, szacuje szum i sprawdza obecność wzoru.
- **Co ciekawe:** Znak powstaje podczas generowania i nie wymaga dodatkowego treningu modelu.

## 4. [Proactive Detection of Voice Cloning with Localized Watermarking (AudioSeal)](https://proceedings.mlr.press/v235/san-roman24a.html)

- **Problem:** Wykrywanie oznaczonej mowy syntetycznej i lokalizowanie jej fragmentów w nagraniu.
- **Algorytm:** Wspólnie trenowane sieci dodają niesłyszalny znak i wykrywają go na poziomie próbek audio. Funkcja straty uwzględnia maskowanie słuchowe oraz lokalizację znaku.
- **Co ciekawe:** Pozwala wskazać konkretne oznaczone fragmenty, nawet gdy połączono je z nieoznaczonym dźwiękiem.

## 5. [A Watermark for Large Language Models](https://proceedings.mlr.press/v202/kirchenbauer23a.html)

- **Problem:** Rozpoznawanie tekstu wygenerowanego przez oznaczony LLM bez dostępu do jego parametrów ani API podczas detekcji.
- **Algorytm:** Dla każdego kroku generowania kontekst poprzedniego tokenu służy do pseudolosowego wyznaczenia „zielonej listy” tokenów. Model zwiększa ich logity przed losowaniem kolejnego tokenu. Detektor odtwarza te listy i testem statystycznym sprawdza, czy udział zielonych tokenów jest większy, niż wynikałoby z przypadku.
- **Co ciekawe:** Praca Kirchenbauera i współautorów (ICML 2023) jasno pokazuje cały proces: osadzanie znaku przez zmianę prawdopodobieństw tokenów, detekcję oraz kompromis między jakością tekstu a wykrywalnością. To dobry punkt wyjścia do zrozumienia nowszych metod znakowania tekstu LLM.

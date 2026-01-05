# Ručne písaný riešič rovníc – 1‑stranové zhrnutie metód

## Cieľ
Cieľom projektu je rozpoznať ručne písané aritmetické výrazy, zrekonštruovať ich do textu a vypočítať výsledok. Systém je postavený ako pipeline: segmentácia symbolov, klasifikácia, zostavenie výrazu a vyhodnotenie.

## Dáta
Lokálny dataset obsahuje 8 384 symbolov v 15 triedach: číslice 0–9 a operátory +, -, *, /, =. Počty tried sú v rozmedzí 433–655 vzoriek. Pre lepšie pokrytie rôznych rukopisov sú číslice doplnené o MNIST (OpenML).

## Predspracovanie
Každý symbol sa prevedie do odtieňov sivej, invertuje, oreže na aktívny atrament, zmenší na 20×20, vycentruje do 28×28, posunie podľa ťažiska a normalizuje do [0, 1].

## Segmentácia
Najprv sa vytvorí maska atramentu a v osi Y sa nájdu riadky s písaným obsahom. V rámci každého riadku sa symboly rozdelia podľa prázdnych stĺpcov, malé medzery sa vypĺňajú a široké boxy sa delia. Boxy sa na záver mierne rozšíria, aby sa nestratili ťahy.

## Model
Použitý je MLPClassifier s vrstvami (256, 128), aktiváciou ReLU a optimalizátorom Adam. Tréning používa batch size 256, early stopping a maximálne 20 epoch. Baseline bez MNIST dosiahol presnosť 0.935 na symboloch; po doplnení MNIST číslic presnosť stúpla na 0.978.

## Testovanie a výsledky
Na syntetickom teste (200 obrázkov, na každom 1–3 príklady, spolu 404 príkladov typu „číslo operátor číslo =“, vygenerovaných z lokálneho datasetu) dosiahol náš model 0.800 presnosť celého výrazu a 0.804 správny výsledok výpočtu. Presnosť celej snímky (všetky rovnice správne naraz) bola 0.635. Na porovnanie, GPT‑4o dosiahol 0.844 presnosť celého výrazu, 0.849 správny výsledok a 0.73 presnosť celej snímky.

## Výstup
Výstup projektu je webová aplikácia (FastAPI) s interaktívnym kreslením a stránkou so zhrnutím metód a výsledkov. Kód a dáta sú súčasťou repozitára.

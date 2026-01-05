# Ručne písaný riešič rovníc - 1-stranové zhrnutie metód

## Problém

Rozpoznať ručne písané aritmetické výrazy a vyriešiť ich. Systém segmentuje kresbu na symboly, klasifikuje každý symbol (číslice a operátory), zrekonštruuje výraz a vyhodnotí ho.

## Dáta

- Lokálny dataset (uložený v `data/`): 8 384 obrázkov symbolov v 15 triedach (číslice 0-9 plus operátory +, -, *, /, =).
- Extra priečinky `dec/`, `x/`, `y/`, `z/` existujú, ale aktuálny model ich nepoužíva.
- Číslice MNIST: načítané z OpenML cez `fetch_openml` a použité len pre číslice.
- Počty tried (lokálny dataset):

| Trieda | Počet | Trieda | Počet | Trieda | Počet |
| ------ | -----: | ------ | -----: | ------ | -----: |
| 0      |    595 | 5      |    433 | add    |    596 |
| 1      |    562 | 6      |    581 | sub    |    655 |
| 2      |    433 | 7      |    533 | eq     |    634 |
| 3      |    541 | 8      |    554 | div    |    618 |
| 4      |    526 | 9      |    546 | mul    |    577 |

Počty tried sa pohybujú od 433 do 655, takže je to mierna nevyváženosť, nie extrémna.

### Predspracovanie

Každý symbol sa prevedie do odtieňov sivej, invertuje, prahuje, oreže na ohraničenie atramentu, zmenší na 20x20, vycentruje do 28x28, posunie podľa ťažiska a normalizuje na [0, 1]. To zodpovedá vstupu v štýle MNIST.

## Metóda

1) **Segmentácia**: vytvorí sa maska atramentu, potom projekcie riadkov a stĺpcov nájdu pásy rovníc a behy symbolov. Malé medzery sa vyplnia, boxy sa vypchajú a široké boxy sa rozdelia v miestach nízkej hustoty atramentu, aby sa oddelili dotýkajúce symboly.
2) **Klasifikátor**: scikit-learn `MLPClassifier` s vrstvami (256, 128), ReLU, Adam, batch size 256, early stopping a max 20 epoch.
3) **Vyhodnotenie výrazu**: tokeny sa skladajú do čísiel a jednoduchý vyhodnocovač (shunting-yard) vypočíta výsledok a skontroluje rovnosť pri prítomnosti znaku '='.

## Experimenty (pokus -> chyba -> zlepšenie)

**Pokus 1 (baseline)**: Tréning len na lokálnom datasete (číslice + operátory).
**Výsledok**: presnosť 0.935 na stratifikovanom delení 80/20.
**Analýza**: lokálna množina číslic je relatívne malá, čo znižuje rozmanitosť tvarov číslic a zvyšuje zámenu podobných číslic.
**Zlepšenie**: doplnenie číslic o MNIST (OpenML) pri zachovaní lokálnych operátorov.
**Výsledok**: presnosť 0.978 na stratifikovanom delení 80/20.
**Záver**: pridanie MNIST zlepšuje generalizáciu pre číslice bez zmeny dát pre operátory, čo ukazuje, že rozmanitosť dát je dôležitejšia než len ladenie hyperparametrov.

## Výsledky

- Najlepšia presnosť klasifikácie symbolov (MNIST + operátory): **0.978** na stratifikovanom delení 80/20.
- Základná presnosť (len lokálne dáta): **0.935**.
- Model uložený ako `recognizer_mlp.joblib` pre opätovné použitie v aplikácii.

## Analýza zlyhaní

Typické prípady zlyhania (na základe segmentácie a predspracovania):

- **Prekrývajúce sa symboly**: dotýkajúce sa číslice alebo operátory sa môžu zlúčiť do jedného boxu.
- **Bledé ťahy**: fixný prah atramentu môže vynechať slabé ťahy, najmä tenké '-'.
- **Zvislé zápisy**: segmentácia predpokladá prevažne horizontálne rovnice.
- **Tiesne medzery**: nízka hustota atramentu nemusí oddeliť susedné symboly.

Tieto prípady môžu viesť k nesprávnej tokenizácii a chybnému vyhodnoteniu výrazu.

## Budúca práca

- Pridať augmentácie pre operátory (rotácia, hrúbka ťahu, rozmazanie).
- Rozšíriť triedy o `dec`, `x`, `y`, `z` alebo ich explicitne filtrovať.
- Natrénovať malé CNN pre vyššiu robustnosť a porovnať s MLP.
- Vytvoriť end-to-end evaluačnú sadu celých rovníc na meranie presnosti riešenia.
- Prispôsobiť segmentáciu pre viacriadkové alebo zvislé zápisy.

## Reprodukovateľnosť a odovzdanie

Kód a lokálne dáta sú v tomto repozitári. Webová aplikácia beží cez FastAPI (`api/README.md`).
MNIST sa načítava automaticky pri prvom tréningu cez OpenML; ak nechcete sieť, upravte tréning tak, aby používal len `data/`.

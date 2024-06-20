# Julia/Pythonで学ぶ"心で理解(ﾜｶ)る統計力学"

このディレクトリは統計力学1(カノニカル分布まで)を学んだばかりの初学者が統計の定理や原理を頭だけでなく心で理解するために執筆されたものである.

Jupyter Notebookの形式で資料を作っているので、以下JupyterNotebookを開き実行するための環境構築(準備)手順を記す。Juliaで実行する場合とPythonで実行する場合があるのでそれぞれ記すのでどちらか好みの言語の方で環境構築もらえれば良い。(特にこだわりがないのであればJuliaをお勧めする)

<br>


## 1. Juliaの環境構築

### 1-1. Juliaupのインストール

(Mac/Linuxの場合)
```
curl -fsSL https://install.julialang.org | sh
```
をターミナルから実行する。

(Windowsの場合)以下を実行
```
winget install julia -s msstore
```
以下を訊かれるが、そのままEnterを押して`Proceed with installation`を実行すれば良い。

![juliaup_install](./Juliaup_Install.jpeg)

インストール終了後
```
julia
```
実行すればJuliaが立ち上がれば問題ない。(「juliaコマンドが見つからない」というエラーが出た場合はターミナルを一旦閉じてもう一度開いてから実行する)
![julia_activation](./julia_activation.png)

<br>

### 1-2. 必要なライブラリ(パッケージ)のインストール
Juliaを開いている状態で`]`を押してみよう。すると以下のような画面に移行する。
![julia_package](./julia_package.png)
これはパッケージモードであり、ライブラリ(パッケージ)の管理をする際に使用する。`Backspace`キーを押すと元のモードに戻ることが出来る。

今回使用するのは`LinearAlgebra`, `Distributions`, `Plots`, `IJulia`なのでそれらのライブライを以下のコマンドでインストールする。(パッケージモードで実行)
```
add LinearAlgebra Distributions Plots IJulia
```
さらに以下を実行
```
build
```
これでJupyterNotebookでJuliaを使う準備が出来た。


## 2. Pythonのインストール
今回はminiforge(conda)を使ってPythonの環境を構築する。(pipの方が軽量だけど、素人が使う分にはcondaの方が問題が起きづらい。)(AppleSiliconのPC含め安定に動作するのはminiforgeらしいので)
(当然既に自身の環境を持っている人はそれを使っても構いません)

### 2.1 miniforgeのインストール
[githubのminiforgeのサイト](https://github.com/conda-forge/miniforge/?tab=readme-ov-file)から自身のOSにあったインストーラをダウンロード。

ダウンロードしたディレクトリに行って(例えば`cd ~/Downloads`)、ダウンロードしたファイルを実行(下の例はApple SiliconのMac。他の場合はファイル名が違いますので適宜)
```
bash ./Miniforge3-MacOSX-arm64.sh
```
全部yesで答えて、インストールが終わったらターミナルを開き直して、`python`と打ってみる。起動すれば成功。`exit()`で抜け出せる。

デフォルトでcondaのbase環境が自動的にactivateされるようになってしまっているので、
```
conda config --set auto_activate_base false
conda deactivate
```
としておく。

### 2.2 仮想環境の準備
今回の授業用に仮想環境を作っておこう。今回はなんとなくpython-3.10を使うことにする。
```
conda create -n py310_ToK python=3.10 numpy scipy matplotlib jupyter
```
授業用の仮想環境を作り、`conda activate py310_ToK`でアクティベートしておく。



## 3. VSCodeのインストール　
コーディングする際のエディターとしてVSCodeを[このサイト](https://code.visualstudio.com/download)からインストールする。

VSCodeを起動し、四角が４つ集まっているアイコンを押すと、拡張機能を入れることが出来る。
Julia, Jupyterを検索し、インストールする。
![extention_julia](./extention_julia.png)
![extention_jupyter](./extention_jupyter.png)

## 4. Jupyter Notebookの起動
`Ctrl+Shift+P`を押して、`create new jupyter~`と打って、`create new jupyter notebook`を選択するとJupyter Notebookを起動。
![activate](create_new.png)

<br>

以下のような画面が出てくる。`Python`となっているところをJuliaに変更し、カーネルの選択をJuliaのものにする。
![jupyter](jupyter.png)

Pythonを選択した人はカーネルの選択で`Python環境`=>`py310_ToK`を選択。
![step1](step1.png)
![step2](step2.png)

これで準備完了。

"""
共通で使う関数
"""
using JSON
using Plots
using DataFrames
using CSV
#using PyPlot
#import matplotlib.pyplot as plt

function isWhite(a::RGB{N0f8})
    if(a==(255,255,255))
        return true
    else
        return false
    end
end

function copy_img(img_back::Array{RGB{N0f8},2}, img_obj::Array{RGB{N0f8},2}, x::Int, y::Int, isTrans::Bool)
    """
    img_back にimg_objをコピーする

    Parameters
    ----------
    img_back: 3d numpy.ndarray
        背景画像
    img_obj: 3d numpy.ndarray
        コピーする物体の画像
    x, y: int
        img 上で張り付ける座標
    isTrans: bool
        True: 白(255, 255, 255)を透明にする

    Returns
    -------
    img_back2: 3d numpy.ndarray
        コピー後の画像

    """

    # 引数のimg_backとimg_objが書き変わらないようにコピーする (A)
    img_obj2 = copy(img_obj)
    img_back2 = copy(img_back)
    w, h, _ = size(img_obj2)...

    if(isTrans)
        # img_obj2の白領域を透明にする処理 (B)
        # img_obj2の白領域に背景画像をコピーする
        idx = findall(iswhite, img_obj2)
        img_back_rect = copy(img_back[x:x+w, y:y+h])
        img_obj2[idx] = img_back_rect[idx]
    end

    # img_obj2をimg_back2にコピー(C)
    img_back2[x:x+w, y:y+h] = img_obj2
    return img_back2
end


function show_graph(file::String, target_reward, target_step)
    """
    学習曲線の表示

    Parameters
    ----------
    target_reward: float or None
        rewardの目標値に線を引く
    target_step: float or None
        stepの目標値に線を引く
    """
    hist = CSV.read(file*".csv", DataFrame)
    #hist = np.load(pathname + '.npz')
    eval_rwd = hist.eval_rwds#hist['eval_rwds'].tolist()
    eval_step = hist.eval_steps
    eval_x = hist.eval_x

    #plt.figure(figsize=(8, 4))
    #plt.subplots_adjust(hspace=0.6)

    # reward / episode
    #plt.subplot(211)
    p1 = plot(eval_x, eval_rwd, color=:blue, width=2.0, marker=:circle, title="rewards / episode", gridwidth=2.0)
    #plt.plot(eval_x, eval_rwd, 'b.-')
    if(target_reward!==nothing)
        p1 = plot!([eval_x[1], eval_x[length(eval_x)]], [target_reward, target_reward], color =:red, linestyle=:dot)
    end

    # steps / episode
    #plt.subplot(212)
    p2 = plot(eval_x, eval_step, color=:blue, width=2.0, marker=:circle, xlabel = "steps", title="steps / episode", gridwidth=2.0)
    if(target_step!==nothing)
        p1 = plot!([eval_x[1], eval_x[length(eval_x)]], [target_step, target_step], color =:red, linestyle=:dot)
    end
    plot(p1,p2,layout=(1,2),size=(600,300))
end

# 以下グリッドサーチでのみ使用
function save_json(filename::String, data)
    """ dict型変数を json形式で保存 """
    j_data = JSON.json(data)
    f = open(filename, "w")
    #hello.jsonを作成
        
    JSON.print(f, j_data)
end


function load_json(file::String)
    """ json形式のファイルをdict型変数に読み込み """
    data = JSON.parsefile(file)
    return data
end
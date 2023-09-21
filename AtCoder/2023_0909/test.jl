using LinearAlgebra

function potions()
    N, H, X = parse.(Int, split(readline()))
    P = parse.(Int, split(readline()))

    for i in 1:N
        if(H+P[i]>=X)
            println(i)
            return nothing
        end
    end
    return nothing
end

function MN()
    N = parse(Int, readline())
    A = parse.(Int, split(readline()))
    sort!(A)
    for i in 1:length(A)-1
        if(A[i+1]-A[i]>1)
            println(A[i]+1)
            return
        end
    end
    return nothing
end

function LP()
    dict = Dict()
    dict["tourist"] = 3858
    dict["ksun48"] = 3679
    dict["Benq"] = 3658
    dict["Um_nik"] = 3648
    dict["apiad"] = 3638
    dict["Stonefeang"] = 3630
    dict["ecnerwala"] = 3613
    dict["mnbvmar"] = 3555
    dict["newbiedmy"] = 3516
    dict["semiexp"] = 3481

    S = readline()
    println(dict[S])
end

function Mes()
    N = parse(Int, readline())
    print(1)
    for i in 1:N
        sw = true
        for j in 1:9
            if(N%j == 0 && i%div(N,j)==0)
                print(j)
                sw = false
                break
            end
        end
        if(sw)
            print("-")
        end
    end
    println("")
end

function check(i::Int, j::Int, C::Matrix{Int})
    sw = true
    v1 = [C[i, k] for k in 1:3 if(k!=j)]
    v2 = [C[k, j] for k in 1:3 if(k!=i)]
    if(v1[1]==v1[2] && v1[1]>0)
        sw = false
        return sw
    end
    if(v2[1]==v2[2] && v2[1]>0)
        sw = false
        return sw
    end
    if(i==j)
        v3 = [C[k, k] for k in 1:3 if(k!=i)]
        if(v3[1]==v3[2] && v3[1]>0)
            sw = false
            return sw
        end
    end
    if(i+j==4)
        v4 = [C[k, 4-k] for k in 1:3 if(k!=i)]
        if(v4[1]==v4[2] && v4[1]>0)
            sw = false
            return sw
        end
    end
    return sw
end

using Combinatorics

function FH()
    C = zeros(Int, 3, 3)
    for i in 1:3
        C[i,:] = parse.(Int, split(readline()))
    end
    v0 = [i for i in 1:9]
    sum = 0
    ans = 0
    for per in permutations(v0)
        sw = true
        TkM = zeros(Int, 3, 3)
        for k in per
            i = div(k-1, 3)+1
            j = (k-1)%3 + 1
            TkM[i,j] = C[i,j]
            sw = check(i, j, TkM)
            if(sw==false)
                break
            end
        end
        sum += 1
        if(sw)
            ans += 1
        end
    end
    println(ans/sum)
end

function MW()
    N, M = parse.(Int, split(readline()))
    L = parse.(Int, split(readline()))
    lmax = maximum(L)
    lsum = sum(L) + length(L)-1
    wmin = lmax
    wmax = lsum
    sw = true
    w = wmin
    while(sw)
        s = -1
        wl = 1
        for i in 1:N
            s += L[i]+1
            if(s > w)
                wl += 1
                s = L[i]
                if(wl>M)
                    wmin = w
                    break
                end
            end
        end
        if(wl<=M)
            wmax = w
        end
        if(wmax-wmin < 2)
            println(wmax)
            return
        end
        w = div(wmax+wmin,2)
    end
end

function BS()
    N, X, Y = parse.(Int, split(readline()))
    P = zeros(Int, N-1)
    T = zeros(Int, N-1)
    for i in 1:N-1
        P[i],T[i] = parse.(Int, split(readline()))
    end
    Q = parse(Int, readline())
    q = zeros(Int, Q)
    for i in 1:Q
        q[i] = parse(Int, readline())        
    end

    dict = Dict()

    L = 840
    for i in 1:Q
        t0 = q[i] + X
        t0m = t0 % L
        if(haskey(dict, t0m))
            println(t0 + dict[t0m])
        else
            t = t0m
            for j in 1:N-1
                if(t==0)
                    t = T[j]
                else
                    t = t + (P[j]-(t-1)%P[j]-1) + T[j]
                end
            end
            t += Y
            dict[t0m] = t - t0m
            println(t0 + dict[t0m])
        end
    end
end

BS()
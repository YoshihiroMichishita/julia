function SS()
    N = parse(Int, readline())
    S = readline()
    T = readline()
    for it in 1:N
        if(S[it]==T[it]||(S[it],T[it])==('1','l')||(S[it],T[it])==('l','1')||(S[it],T[it])==('0','o')||(S[it],T[it])==('o','0'))
            continue
        else
            println("No")
            return
        end
    end
    println("Yes")
end

function Dic()
    N, M = parse.(Int, split(readline()))
    a = zeros(Int, N, M)
    for it in 1:M
        a[:,it] = parse.(Int, split(readline()))
    end
    fr = zeros(Int, N, N)
    for it in 1:M
        for nn in 1:N-1
            fr[a[nn,it],a[nn+1,it]] += 1
            fr[a[nn+1,it],a[nn,it]] += 1
        end
    end
    #println(fr)
    ans = 0
    for p in 1:N-1
        for q in p+1:N
            if(fr[p,q]==0)
                ans += 1
            end
        end
    end
    println(ans)
end

function Dash()
    N, M, H, K = parse.(Int, split(readline()))
    S = readline()
    rec = []
    for it in 1:M
        x,y = parse.(Int, split(readline()))
        push!(rec, [x,y])
    end
    now = [0, 0]
    her = H
    for it in 1:M
        s = S[it]
        if(s=='U')
            now += [0, 1]
        elseif(s=='D')
            now += [0, -1]
        elseif(s=='L')
            now += [-1, 0]
        elseif(s=='R')
            now += [1, 0]
        end
        her -= 1
        if(her<0)
            println("No")
            return
        end
        if(isempty(findall(x->x==now, rec)))
        elseif(her<K)
            her = K
        end
    end
    println("Yes")
end


Dash()

function MH()
    N, M, K = parse.(Int, split(readline()))
    S = readline()
    c = min(N, K)
end

function OW()
    N = parse(Int, readline())
    S = readline()
    score = 0
    name = "atcoder"
    for it in 1:N
        if(S[it]=='T')
            score += 1
            if(score==0)
                name = "A"
            end
        else
            score -= 1
            if(score==0)
                name = "T"
            end
        end
        
    end
    if(score>0)
        println("T")
    elseif(score<0)
        println("A")
    else
        println(name)
    end
end

function FtG()
    N = parse(Int, readline())
    A = parse.(Int, split(readline()))
    ans::Vector{Int} = []
    for it in 1:N-1
        if(abs(A[it]-A[it+1])==1)
            push!(ans, A[it])
        else
            a = A[it]
            if(A[it]>A[it+1])
                while(a>A[it+1])
                    push!(ans, a)
                    a -= 1
                end
            else
                while(a<A[it+1])
                    push!(ans, a)
                    a += 1
                end
            end
        end
    end
    push!(ans, A[N])
    for l in ans
        print("$(l) ")
    end
end

function ACC()
    S = readline()
    T = readline()
    count = zeros(Int, 28)

    for l in S
        if(l=='@')
            count[27] += 1
        else
            count[Int(l-'a'+1)] +=1
        end
    end
    for l in T
        if(l=='@')
            count[28] += 1
        else
            count[Int(l-'a'+1)] -=1
        end
    end

    #println(count)
    for it in 1:26
        if(count[it]==0)
        else
            if(it==1 || it==Int('c'-'a'+1) || it==Int('d'-'a'+1) || it==Int('e'-'a'+1) || it==Int('o'-'a'+1) || it==Int('r'-'a'+1) || it==Int('t'-'a'+1))
                if(count[it]>0)
                    count[28]-=count[it]
                    if(count[28]<0)
                        println("No")
                        return nothing
                    end
                else
                    count[27]+=count[it]
                    if(count[27]<0)
                        println("No")
                        return nothing
                    end
                end
            else
                println("No")
                return nothing
            end
        end
    end
    println("Yes")

end

function BitM()
    S0 = readline()
    S = collect(S0)
    N = parse(Int, readline())
    dig = Int(round(log2(N)))
    T = zeros(Int, dig+1)
    for it in 0:dig
        d = 2^(dig-it)
        s = div(N, d)
        N -= s * d
        T[it+1] = s   
    end
    #println(T)
    L = length(S)
    if dig > L
        ans = 0
        for it in 1:L 
            if S[it] == '0'
            else
                ans += 2^(it-1)
            end
        end
        println(ans)    
    else
        dis = L - dig
        if dis > 0
            for ss in S[1:dis]
                if ss == '1'
                    println(-1)
                    return nothing
                end
            end
        end
        mark = 0
        for it in 1:dig
            if S[dis+it] == '1'
                T[it] -= 1
            end
            if T[it] > 0
                mark = it
            elseif T[it] < 0
                if mark == 0
                    println(-1)
                    return nothing
                end
                T[mark] = 0
                T[mark+1:end] .= 1
                break
            end
        end
        #println(T)
        ans = 0
        sw = false
        for it in 1:dig
            if (S[dis+it] == '?')
                if(sw)
                    S[dis+it] = '1'
                else
                    S[dis+it] = "$(T[it])"[1]
                    #println(T[it])
                end
                #println(S[dis+it])
            elseif(T[it]==1)
                sw = true 
            end
            
            a = parse(Int, S[dis+it])
            ans += a*2^(dig-it)
        end
        println(ans)
    end
end

BitM()

#=
function BitM()
    S = readline()
    N = parse(Int, readline())
    dig = Int(round(Int, log2(N)))
    T = zeros(Int, dig+1)
    for it in 0:dig
        d = 2^(dig-it)
        s = div(N, d)
        N -= s* d
        T[it+1] = s   
    end
    println(T)
    L = length(S)
    if(dig>L)
        ans = 0
        for it in 1:L 
            if(S[it] == '0')
            else
                ans += 2^(it-1)
            end
        end
        println(ans)    
    else
        dis = L - dig
        if(dis>0)
            for ss in S[1:dis]
                if(ss=='1')
                    println(-1)
                    return nothing
                end
            end
        end
        mark = 0
        for it in 1:dig
            if(S[dis+it] == '1')
                T[it] -=1
            end
            if(T[it]>0)
                mark = it
            elseif(T[it]<0)
                if(mark==0)
                    println(-1)
                    return nothing
                end
                T[mark] = 0
                T[mark+1:end] .= 1
                break
            end
        end
        println(T)
        ans = 0
        println("S:$(typeof(S)), dis:$(typeof(dis))")
        for it in 1:dig
            if(S[dis+it] == '?')
                println("S:$(typeof(S[dis+it])), dis:$(typeof(dis)), it:$(typeof(it)), 1:$(typeof('1'))")
                S[dis+it] = '1'
            end
            a = parse(Int, S[dis+it])
            ans += a*2^(dig-it+1)
        end
        println(ans)

    end
end

BitM()=#
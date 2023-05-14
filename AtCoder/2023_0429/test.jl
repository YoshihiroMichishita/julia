function NcQ()
    N, A, B = parse.(Int, split(readline()))
    C = parse.(Int, split(readline()))
    for i in 1:N
        if(C[i]-A == B)
            println(i)
            return nothing
        end        
    end
end

function Shift_Check(i, j, A, B)
    A_rshift = circshift(A, (i, 0))
    B_cshift = circshift(B, (0, j))
    return A_rshift == B_cshift
end

function SMiRPG()
    H, W = parse.(Int, split(readline()))
    #A = zeros(Char, H, W)
    A = fill(' ', H, W)
    #B = zeros(Char, H, W)
    B = fill(' ', H, W)
    for i in 1:H
        R = readline()
        for j in 1:W
            A[i,j] = R[j]
        end
    end
    for i in 1:H
        R = readline()
        for j in 1:W
            B[i,j] = R[j]
        end
    end
    for i in 1:H, j in 1:W
        sw = Shift_Check(i,j,A,B)
        if(sw)
            println("Yes")
            return nothing
        end      
    end
    println("No")
end

function Cross()
    H, W = parse.(Int, split(readline()))
    N = min(H,W)
    S = zeros(Int, N)
    C = fill(' ', H, W)
    for i in 1:H
        R = readline()
        for j in 1:W
            C[i,j] = R[j]
        end
    end
    #println(C)
    list = []
    filt = ['#' '.' '#'; '.' '#' '.'; '#' '.' '#']
    for i in 2:H-1, j in 2:W-1
        Z = C[i-1:i+1, j-1:j+1]
        if(Z == filt)
            S[1]+=1
            push!(list, (i,j))
        end
    end

    for ss in 2:N
        l = size(list)[1]
        #filt2 = fill('.', 2ss+1, 2ss+1)
        #for d in 1:2ss+1
        #    filt2[d,d] = '#'
        #    filt2[d,2ss+2-d] = '#'
        #end
        for i in 1:l
            X, Y = popfirst!(list)
            try
                if(C[X-ss,Y-ss]=='#' && C[X+ss,Y-ss]=='#' && C[X-ss,Y+ss]=='#' && C[X+ss,Y+ss]=='#')
                    S[ss]+=1
                    S[ss-1]-=1
                    push!(list, (X,Y))
                end
            catch
            end
        end
    end
    for i in 1:N
        print("$(S[i]) ")        
    end
    println("")
end

function find_prime(M)
    prime = [2, 3, 5]
    for it in 5:M
        sw = true
        for d in prime
            if(it%d == 0)
                sw = false
                break
            end
        end
        if(sw)
            push!(prime, it)
        end
    end
    return prime
end

function AABCC()
    N = parse(Int, readline())
    M = Int(round(sqrt(N/12)))
    prime = find_prime(M)
    L = length(prime)
    #println(L)
    count = 0
    for c in 3:L
        C= prime[c]
        for a in 1:c-2
            A = prime[a]
            if(A^2*(prime[a+1])*C^2>N)
                break
            end
            for b in a+1:c-1
                B = prime[b]
                if(A^2*B*C^2>N)
                    break
                else
                    #println("A:$(A), B:$(B), C:$(C)")
                    count += 1
                end
            end
        end        
    end
    println(count)
end

#=
function ModInt(m,n)
    L = 998244353
    for i in 1:n-1
        m += L
        if(m%n==0)
             return div(m,n)
        end
    end
    return 0
end

function PrimeDiv(N)
    #2~6
    divpow = zeros(Int, 3)
    m= [2,3,5]
    for it in 1:3
        while(N%m[it]==0)
            N = div(N, m[it])
            divpow[it]+=1
        end
    end
    return divpow
end

function DP3()
    N = parse(Int, readline())
    pd = PrimeDiv(N)
    #println(pd)
    if(sum(pd)==0)
        println(0)
        return 0
    end
    dict = Dict{Int, Rational{Int64}}()
    for a2 in 0:pd[1]
        for a3 in 0:pd[2]
            for a5 in 0:pd[3]
                A = 2^a2*3^a3*5^a5
                dict[A] = 0
            end
        end
    end
    dict[1] = 1
    que = [1]
    while(!isempty(que)) 
        s = popfirst!(que)
        for x in 2:6
            p = s*x
            if(haskey(dict, p))
                if(dict[p]==0)
                    push!(que, p)
                end
                dict[p] += dict[s]//5
                if(denominator(dict[p])>998244353)
                    dict[p] = numerator(dict[p])//(denominator(dict[p])%998244353)
                end
                if(numerator(dict[p])>998244353)
                    dict[p] = (numerator(dict[p])%998244353)//denominator(dict[p])
                end
            end
        end
    end
    ans = ModInt(numerator(dict[N]), denominator(dict[N]))
    println(ans)
end
=#
function modinv(a, m)
    # aとmが互いに素であることを前提とする
    b = BigInt(a)
    e = BigInt(m-2)
    r = 1
    while e > 0
        if isodd(e)
            r = (r * b) % m
        end
        b = (b * b) % m
        e = e ÷ 2
    end
    return r
end

function ModInt(m,n)
    L = 998244353
    r = modinv(n,L)
    x = (r*m)%L
    return x
end

function PrimeDiv(N)
    #2~6
    divpow = zeros(Int, 3)
    m= [2,3,5]
    for it in 1:3
        while N % m[it] == 0
            N = N ÷ m[it]
            divpow[it] += 1
        end
    end
    ans = false
    if(N>1)
        ans = true
    end
    return divpow, ans
end

function DP3()
    N = parse(Int, readline())
    pd, sw = PrimeDiv(N)
    if(sw)
        println(0)
        return 0
    end
    dict = Dict{Int, Rational{Int64}}()
    for a2 in 0:pd[1], a3 in 0:pd[2], a5 in 0:pd[3]
        A = 2^a2 * 3^a3 * 5^a5
        dict[A] = 0
    end
    dict[1] = 1
    que = [1]
    qstart = 1
    qend = 1
    while qstart <= qend
        s = que[qstart]
        qstart += 1
        for x in 2:6
            p = s * x
            if(!haskey(dict, p))
                continue
            end
            if dict[p] == 0
                qend += 1
                push!(que,p)
            end
            dict[p] += dict[s] // 5
            if(dict[p].den>998244353)
                dict[p] = dict[p].num//(dict[p].den%998244353)
            end
            if(dict[p].num>998244353)
                dict[p] = (dict[p].num%998244353)//dict[p].den
            end
            #=
            if dict[p].den > 998244353
                dict[p].num, dict[p].den = dict[p].num ÷ (dict[p].den mod 998244353), 998244353
            end
            if dict[p].num > 998244353
                dict[p].num = dict[p].num mod 998244353
            end=#
        end
    end
    ans = ModInt(dict[N].num, dict[N].den)
    println(ans)
end

#dpでk個xをoに変えた時の最長連続数を記録、端を跨いだらc=1
function find_max(S)
    L = length(S)
    K=0
    for it in 1:L
        if(S[it]=='x')
            K +=1
        end
    end
    max_len_k = zeros(Int, K)
    max_len_per = zeros(Int, K)
    test = zeros(Int, K)
    k_it=0
    for it in 1:L
        if(S[it]=='x')
            k_it
        end
    end
end

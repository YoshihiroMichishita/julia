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
        while(N%m==0)
            N = div(N, m[it])
            divpow[it]+=1
        end
    end
    return divpow
end

function DP3()
    N = parse(Int, readline())
    pd = PrimeDiv(N)
    if(sum(pd)==0)
        return 0
    end
    dic = Dict()
    for a2 in 0:pd[1]
        for a3 in 0:pd[2]
            for a5 in 0:pd[3]
                A = 2^a2*3^a3*5^a5
                dict["$(A)"] = 0
            end
        end
    end
    dict["1"] = 1
    S = sum(pd)
    que = [1]
    while(isempty(que)) 
        s = pop!(first)
    end
end


AABCC()
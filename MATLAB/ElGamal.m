classdef ElGamal
    properties
        bitLength = 0;
        p = 0;
        q = 0;
        g = 0;
        h = 0;
        s = 0;
    end

    methods
        function obj = ElGamal(bitLength)
            obj.bitLength = bitLength;
            [obj.p,obj.q,obj.g,obj.h,obj.s] = obj.KeyGen(obj.bitLength);
        end

        function [p,q,g,h,s] = KeyGen(~,bitLength)
            p = getSafePrime(bitLength);
            q = (p-1)/2;
            g = getGenerator(p);
            % s = randi([0,q-1]); % s=0 is unsecure.
            s = randi([1,q-1]);
            h = modPow(g,s,p);
        end

        function m = Encode(obj,x,gamma)
            m = round(x*gamma);
            firstDecimalPlace = modMult(x*gamma,10,10);

            if m < 0
                if m < -(obj.q+1)
                    error('Underflow.')
                else
                    m = m+obj.p;
                end
            else
                if m >= obj.q
                    error('Overflow')
                end
            end
    
            if isElement(m,obj.p)
                return
            else
                if firstDecimalPlace >= 5 || firstDecimalPlace == 0
                    for i = 1:obj.q
                        if isElement(m-i,obj.p)
                            m = m-i;
                            return
                        elseif isElement(m+i,obj.p)
                            m = m+i;
                            return
                        end
                    end
                else
                    for i = 1:obj.q
                        if isElement(m+i,obj.p)
                            m = m+i;
                            return
                        elseif isElement(m-i,obj.p)
                            m = m-i;
                            return
                        end
                    end
                end
            end

            error('Failed to encode.')
        end

        function x = Decode(obj,m,gamma)
            if m >= obj.q
                x = (m-obj.p)/gamma;
            else
                x = m/gamma;
            end
        end

        function c = Encrypt(obj,m)
            r = randi([1,obj.q-1]);
            c(1) = modPow(obj.g,r,obj.p);
            c(2) = modMult(m,modPow(obj.h,r,obj.p),obj.p);
        end

        function m = Decrypt(obj,c)
            m = modMult(c(2),modInv(modPow(c(1),obj.s,obj.p),obj.p),obj.p);
        end

        function c = Enc(obj,x,gamma)
            c = obj.Encrypt(obj.Encode(x,gamma));
        end

        function x = Dec(obj,c,gamma)
            x = obj.Decode(obj.Decrypt(c),gamma);
        end

        function c = Mult(obj,a,b)
            c(1) = modMult(a(1),b(1),obj.p);
            c(2) = modMult(a(2),b(2),obj.p);
        end
    end
end

function p = getSafePrime(bitLength)
    binMin = bitsll(1,bitLength-1);
    binMax = bitsll(binMin,1);

    q = randi([binMin,binMax-1]);
    q = bitset(q,1);
    q = bitset(q,bitLength);
    
    while ~isprime(q) || ~isprime(2*q+1)
        q = randi([binMin,binMax-1]);
        q = bitset(q,1);
        q = bitset(q,bitLength);
    end
    
    p = 2*q+1;
end

function g = getGenerator(p)
    q = (p-1)/2;

    g = 2;
    while ~isGenerator(g,q,p)
        g = g+1;
    end
    
    if g >= p
        error('There is no generator.')
    end
end

function boolean = isGenerator(g,q,p)
    if modPow(g,q,p) == 1
        boolean = true;
    else
        boolean = false;
    end
end

function boolean = isElement(m,p)
    q = (p-1)/2;
    if modPow(m,q,p) == 1
        boolean = true;
    else
        boolean = false;
    end
end

function b = Mod(a,m) 
    b = mod(a,m);

    if b < 0
        b = b+m;
    end
end

function c = modAdd(a,b,m) 
    c = Mod(a+b,m);
end

function c = modMult(a,b,m)
    c = 0;
    
    if a < 0
        a = a+m;
    end

    % bitwise operation
    while b >= 1
        if mod(b,2) == 1
            c = mod(a+c,m);
        end
        a = mod(a*2,m);
        b = floor(b/2);
    end
end

function c = modPow(a,b,m)
    if a < 0
        a = a+m;
    end

    % Bruce Schneier's algorithm
    c = 1;
    while b >= 1
        if mod(b,2) == 1
            c = modMult(a,c,m);
        end
        a = modMult(a,a,m);
        b = floor(b/2);
    end
end

function u = modInv(a,m)
    % extended Euclidean algorithm
    b = m;
    u = 1;
    v = 0;
    
    while b ~= 0
        t = floor(a/b);
        a = a-t*b;
        u = u-t*v;
        [a,b] = swap(a,b);
        [u,v] = swap(u,v);
    end
    
    if u < 0
        u = u+m;
    end
end

function [x,y] = swap(a,b)
    x = b;
    y = a;
end

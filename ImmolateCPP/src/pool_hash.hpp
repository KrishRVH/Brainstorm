/*
 * Pool Hash - SHA-256 computation for pool snapshots
 * Provides reproducible pool IDs for debugging mismatches
 */

#ifndef POOL_HASH_HPP
#define POOL_HASH_HPP

#include <cstdint>
#include <cstring>
#include <string>

// Simple SHA-256 implementation (can be replaced with OpenSSL if available)
class SHA256 {
private:
    static const uint32_t K[64];
    uint32_t H[8];
    uint8_t data[64];
    uint32_t datalen;
    uint64_t bitlen;
    
    static uint32_t rotr(uint32_t x, uint32_t n) {
        return (x >> n) | (x << (32 - n));
    }
    
    static uint32_t choose(uint32_t e, uint32_t f, uint32_t g) {
        return (e & f) ^ ((~e) & g);
    }
    
    static uint32_t majority(uint32_t a, uint32_t b, uint32_t c) {
        return (a & b) ^ (a & c) ^ (b & c);
    }
    
    static uint32_t sig0(uint32_t x) {
        return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
    }
    
    static uint32_t sig1(uint32_t x) {
        return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
    }
    
    void transform() {
        uint32_t W[64];
        uint32_t a, b, c, d, e, f, g, h;
        uint32_t t1, t2;
        
        for (int i = 0; i < 16; i++) {
            W[i] = (data[i * 4] << 24) | (data[i * 4 + 1] << 16) |
                   (data[i * 4 + 2] << 8) | data[i * 4 + 3];
        }
        
        for (int i = 16; i < 64; i++) {
            W[i] = sig1(W[i - 2]) + W[i - 7] + sig0(W[i - 15]) + W[i - 16];
        }
        
        a = H[0]; b = H[1]; c = H[2]; d = H[3];
        e = H[4]; f = H[5]; g = H[6]; h = H[7];
        
        for (int i = 0; i < 64; i++) {
            t1 = h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + 
                 choose(e, f, g) + K[i] + W[i];
            t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + majority(a, b, c);
            h = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        
        H[0] += a; H[1] += b; H[2] += c; H[3] += d;
        H[4] += e; H[5] += f; H[6] += g; H[7] += h;
    }
    
public:
    SHA256() { reset(); }
    
    void reset() {
        H[0] = 0x6a09e667; H[1] = 0xbb67ae85;
        H[2] = 0x3c6ef372; H[3] = 0xa54ff53a;
        H[4] = 0x510e527f; H[5] = 0x9b05688c;
        H[6] = 0x1f83d9ab; H[7] = 0x5be0cd19;
        datalen = 0;
        bitlen = 0;
    }
    
    void update(const uint8_t* data_in, size_t len) {
        for (size_t i = 0; i < len; i++) {
            data[datalen] = data_in[i];
            datalen++;
            if (datalen == 64) {
                transform();
                bitlen += 512;
                datalen = 0;
            }
        }
    }
    
    void final(uint8_t hash[32]) {
        uint32_t i = datalen;
        
        if (datalen < 56) {
            data[i++] = 0x80;
            while (i < 56) data[i++] = 0x00;
        } else {
            data[i++] = 0x80;
            while (i < 64) data[i++] = 0x00;
            transform();
            memset(data, 0, 56);
        }
        
        bitlen += datalen * 8;
        data[63] = bitlen;
        data[62] = bitlen >> 8;
        data[61] = bitlen >> 16;
        data[60] = bitlen >> 24;
        data[59] = bitlen >> 32;
        data[58] = bitlen >> 40;
        data[57] = bitlen >> 48;
        data[56] = bitlen >> 56;
        transform();
        
        for (i = 0; i < 4; i++) {
            hash[i]      = (H[0] >> (24 - i * 8)) & 0x000000ff;
            hash[i + 4]  = (H[1] >> (24 - i * 8)) & 0x000000ff;
            hash[i + 8]  = (H[2] >> (24 - i * 8)) & 0x000000ff;
            hash[i + 12] = (H[3] >> (24 - i * 8)) & 0x000000ff;
            hash[i + 16] = (H[4] >> (24 - i * 8)) & 0x000000ff;
            hash[i + 20] = (H[5] >> (24 - i * 8)) & 0x000000ff;
            hash[i + 24] = (H[6] >> (24 - i * 8)) & 0x000000ff;
            hash[i + 28] = (H[7] >> (24 - i * 8)) & 0x000000ff;
        }
    }
    
    std::string hexdigest() {
        uint8_t hash[32];
        final(hash);
        
        static const char* hex = "0123456789abcdef";
        std::string result;
        result.reserve(64);
        for (int i = 0; i < 32; i++) {
            result.push_back(hex[(hash[i] >> 4) & 0xF]);
            result.push_back(hex[hash[i] & 0xF]);
        }
        return result;
    }
};

// SHA-256 K constants
const uint32_t SHA256::K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// Helper function to compute pool ID
static std::string compute_pool_id(const uint8_t* data, size_t size) {
    SHA256 sha;
    sha.update(data, size);
    return sha.hexdigest();
}

#endif // POOL_HASH_HPP
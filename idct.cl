void fast_idct(short *vec, int rc, bool by_row);

void fast_idct(short *vec, int rc, bool by_row){
    double a = 3.141592653 / 16.0;

    double c1 = 0.707107;
    double c2 = 0.707107;
    double c3 = 0.980785;
    double c4 = 0.19509;
    double c5 = 0.92388;
    double c6 = 0.382683;
    double c7 = 0.83147;
    double c8 = 0.55557;
    double c9 = 1.41421;

    int si1[8] = {4, 0, 1, 7, 2, 6, 3, 5};
    double s1[8];

    if(by_row) {
        for (int i = 0; i < 8; i++) {
            s1[i] = (double)(vec[rc * 8 + si1[i]] * 8);
        }
    } else {
        for (int i = 0; i < 8; i++) {
            s1[i] = (double)(vec[si1[i] * 8 + rc] * 8);
        }
    }

    double s2[8];
    s2[0] = (c1 * s1[0] + c2 * s1[1]) * 2.0;
    s2[1] = (c5 * s1[4] + c6 * s1[5]) * 2.0;
    s2[2] = (c1 * s1[1] - c2 * s1[0]) * 2.0;
    s2[3] = (c5 * s1[5] - c6 * s1[4]) * 2.0;
    s2[4] = (c3 * s1[2] + c4 * s1[3]) * 2.0;
    s2[5] = (c7 * s1[6] + c8 * s1[7]) * 2.0;
    s2[6] = (c3 * s1[3] - c4 * s1[2]) * 2.0;
    s2[7] = (c7 * s1[7] - c8 * s1[6]) * 2.0;

    double s3[8];
    s3[0] = (s2[4] + s2[5]) * 0.5;
    s3[1] = (s2[6] - s2[7]) * 0.5;
    s3[2] = s2[0];
    s3[3] = (s2[4] - s2[5]) * 0.5;
    s3[4] = s2[1];
    s3[5] = s2[2];
    s3[6] = s2[3];
    s3[7] = (s2[6] + s2[7]) * 0.5;

    double s4[8];
    s4[0] = s3[0];
    s4[1] = s3[1];
    s4[2] = s3[2];
    s4[3] = s3[4];
    s4[4] = s3[5];
    s4[5] = s3[3] * c9;
    s4[6] = s3[6];
    s4[7] = s3[7] * c9;

    double s5[8];
    s5[0] = (s4[2] + s4[3]) * 0.5;
    s5[1] = s4[0];
    s5[2] = (s4[4] + s4[5]) * 0.5;
    s5[3] = (s4[6] + s4[7]) * 0.5;
    s5[4] = (s4[2] - s4[3]) * 0.5;
    s5[5] = s4[1];
    s5[6] = (s4[4] - s4[5]) * 0.5;
    s5[7] = (s4[6] - s4[7]) * 0.5;

    double d[8];
    d[0] = (s5[0] + s5[1]) * 0.5;
    d[1] = (s5[2] - s5[3]) * 0.5;
    d[2] = (s5[2] + s5[3]) * 0.5;
    d[3] = (s5[4] - s5[5]) * 0.5;
    d[4] = (s5[4] + s5[5]) * 0.5;
    d[5] = (s5[6] + s5[7]) * 0.5;
    d[6] = (s5[6] - s5[7]) * 0.5;
    d[7] = (s5[0] - s5[1]) * 0.5;

    if(by_row) {
        for(int i = 0; i < 8; i++){
            if(d[i] < 0) {
                vec[rc * 8 + i] = (short)((d[i] - 4) / 8);
            } else {
                vec[rc * 8 + i] = (short)((d[i] + 4) / 8);
            }
        }
    }
    else {
        for(int i = 0; i < 8; i++){
            if(d[i] < 0){
                vec[i * 8 + rc] = (short)((d[i] - 4) / 8);
            } else {
                vec[i * 8 + rc] = (short)((d[i] + 4) / 8);
            }
        }
    }
}

__kernel void idct_gpu(__global short *image,__global short *det,unsigned int blocks)
{
    size_t index=get_global_id(0);
    if(index*64<blocks)
    {
        char zzi[64] = {  0,  1,  5,  6, 14, 15, 27, 28,
                     2,  4,  7, 13, 16, 26, 29, 42, 
                     3,  8, 12, 17, 25, 30, 41, 43,
                     9, 11, 18, 24, 31, 40, 44, 53,
                    10, 19, 23, 32, 39, 45, 52, 54,
                    20, 22, 33, 38, 46, 51, 55, 60,
                    21, 34, 37, 47, 50, 56, 59, 61,
                    35, 36, 48, 49, 57, 58, 62, 63};

        short uzz_block[64];
        for (int k = 0; k < 64; k++) {
             uzz_block[k] = image[index*64+zzi[k]];
             }
                /* fast inverse discrete cosine transform */
                for (int r = 0; r < 8; r++) {
                    fast_idct(uzz_block, r, true);
                }
                for (int c = 0; c < 8; c++) {
                    fast_idct(uzz_block, c, false);
                }
    
            /* assign back to image */
                for (int k = 0; k < 64; k++) {
                    det[index*64+k] = uzz_block[k];
                }
    }
}
__kernel void to_rgb(__global short *src,__global short *det,unsigned int height,unsigned int width)
{
    size_t index_y=get_global_id(0);
    size_t index_x=get_global_id(1);

    //if(index_y <(height-1)*3 &&index_x<(width-1)*3)
   // {
        short y=(short) (src[0+index_y*width+index_x]);
        short cb=(short) (src[1*width*height+index_y*width+index_x]);
        short cr=(short) (src[2*width*height+index_y*width+index_x]);
      
        short r = (short)(y + ((1.402 * cr) +128));
        short g =(short) (y - ((0.34414 * cb) - (0.71414 * cr) )+128);
        short b =(short) (y + ((1.772 * cb) + 128));

        det[0+index_y*width+index_x] = (r < 0) ? 0 : (r > 255) ? 255 : r;
        det[1*width*height+index_y*width+index_x] = (g < 0) ? 0 : (g > 255) ? 255 : g;
        det[2*width*height+index_y*width+index_x] = (b < 0) ? 0 : (b > 255) ? 255 : b;

   // }
}
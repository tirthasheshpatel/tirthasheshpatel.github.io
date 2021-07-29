Title: Writing code in Pelican!
Date: 2021-07-27 11:30
Category: Code
Tags: coding, c
Author: Will Smith
Summary: Coding in C and looking at code blocks in Pelican!

## Hello world!

This is how you write hello world in C!

```c
#include <stdio.h>

int main(int argc, char **argv)
{
    printf("Hello, world!");
    return 0;
}
```

## Random Number Generation in C

This is how to use the UNU.RAN library to generate random numbers from non-standard distributions!

```c
#include <unuran.h>

#define NUM_SAMPLES 10

int main(int argc, char **argv)
{
    int i;
    double x[NUM_SAMPLES];
    UNUR_DISTR *distr;
    UNUR_PAR   *par;
    UNUR_GEN   *gen;

    distr = unur_distr_normal(NULL, 0);

    par = unur_arou_new();

    gen = unur_init(par);
    if ( gen == NULL ) {
        fprintf(stderr, "falied to initialize the generator!\n");
        exit( EXIT_FAILURE );
    }
    unur_distr_free(distr);

    for (i=0 ; i<NUM_SAMPLES ; ++i) {
        x[i] = unur_sample_cont(gen);
        printf("%lf ", x[i]);
    }
    printf("\n");

    return 0;
}
```

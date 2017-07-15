# enigma-cuda
### about

**enigma-cuda** is a command-line tool for ciphertext-only cryptanalysis of the 
messages encrypted with a Heer, M3 or M4 [Enigma machine][1] that was used by 
Germany in World War II. 

A significant speed improvement is achieved by doing the computations on 
a [GPU processor][6] of a video card. For best results, [Nvidia GTX 1070][8]
or better is recommended.

The algorithm is similar to the one implemented in the [enigma-suite][2] 
software developed by Stefan Krah and used in the
[Enigma@Home][3] project, with the improvements described [here][4] and the
new [partial exhaustion][5] method described by Olaf Ostwald and Frode Weierud.

If compiled with a dynamically linked runtime library, the program requires
installation of [Visual C++ Redistributable for Visual Studio 2015][9]



### command-line options

The command line parameters are compatible with the [enigma-suite][2] program, 
however, some non-essential options are ignored, and some new options are
supported.

**usage:**

`enigma-cuda.exe <options> <trigram file name> <uni/bigram file name> <ciphertext file name>`

**options:**

`-h`             show help

`-v`             show version number

`-R`             resume operation using the state saved in the *00hc.resume*
file. The file is saved in the current working directory, so its location 
is sometimes unpredictable. 
This behavior is preserved for compatibility with enigma-suite.
The format of the file is described in the *enigma.txt* document included with 
the enigma-suite [source code][7]. Note that this file does not save information
about the `-g`, `-e`, `-E` and `-s` settings.

**Edit:** The `-g` option is now stored in the Resume file which breaks 
compatibility with the original enigma-suite format.

`-o <file_name>` save output to file

`-M <model>`     Enigma model : `H`, `M3` or `M4`. If not specified, the model 
is inferred from the keys

`-f <key>`       first key. The format of the key is the same as in enigma-suite.

`-t <key>`       last key. The program tries all keys in the range from first 
key to last key. If both keys are omitted, all possible keys are tried. 
Since computations are performed in blocks of 26^3 keys, the last
three letters in the first and last keys are ignored and assumed to be
AAA to ZZZ. The only exception to this is the case when the first and last keys 
are exactly the same, then only that specific key is tried.

`-x`             turnover mode. Try the keys that result in a left hand wheel
turnover within the message. The default behavior is the opposite, only the keys
without a turnover are tried.

`-a`             turnover mode. Try both keys with and without a left hand wheel 
turnover. The "duplicate" keys that produce the same plain text as some other 
key are still ignored in this mode

`-n <count>`     the number of passes to make, 1 by default

`-z <score>`     stop when this score is reached

`-s <plugs>`     known plugboard connections. E.g., `ABXY` defines the A-B and 
X-Y plugs. The program will not touch these plugs during hill climbing. 
Useful when  some plugboard connections are already known, e.g., from running a 
bombe

`-e <letters>`   [exhaustive search][5] with a single fixed plug. E.g., `ENRXSI`
tells the program to try the E-A..E-Z, N-A..N-Z, ... I-A..I-Z plugs. For 6 
letters the total number of fixed plugs that will be tried is 141.

`-E <letters>`   exhaustive search with multiple fixed plugs. Currently only two
plugs at a time are allowed, E.g., with the `EN` argument the program will try
625 plug pairs, EANB to EZNY


`-p`             start with a random swapping order. Withoug this option, the 
swapping order in the first pass is determined by the letter frequencies in the 
ciphertext

`-g <scoring functions>`    use scoring functions: `0` = IC, `1` = unigrams, 
`2` = bigrams, `3` = trigrams. 
The default is `023`. 

For compatibility with enigma-suite,
the program can use only the unigrams or bigrams file, but not both.

`-d <device number>`   Use the specified GPU device. Device indices start 
with 0. If this switch is not present, the GPU with the highest number of
multiprocessors is used.

`-icuwrmk`       these options of enigma-suite are not supported by 
**enigma-cuda**, but their inclusion on the command line does not cause an error



### feedback
Please send your feedback to
![](http://dxatlas.com/Img/EmailMe.gif)

Alex Shovkoplyas VE3NEA


[1]: https://en.wikipedia.org/wiki/Enigma_machine
[2]: http://www.bytereef.org/enigma-suite.html
[3]: http://www.enigmaathome.net/
[4]: http://www.enigmaathome.net/forum_thread.php?id=814#4143
[5]: http://cryptocellar.org/pubs/Enigma_ModernBreaking.pdf
[6]: http://www.nvidia.ca/object/what-is-gpu-computing.html
[7]: http://www.bytereef.org/enigma-suite.html
[8]: https://www.nvidia.com/en-us/geforce/products/10series/geforce-gtx-1070/
[9]: https://www.microsoft.com/en-us/download/details.aspx?id=48145
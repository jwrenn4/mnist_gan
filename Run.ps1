param ($script, $imagedir, $modeldir)

$epochsize = 256
$numepochs = 100

for ($num=0; $num -lt 11; $num++){
    write-output 'Training GAN to create number $num'
    python $script $num $numepochs --epoch-size $epochsize -i $imagedir -m $modeldir
    }

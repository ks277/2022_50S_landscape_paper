#!/bin/bash

module load chimerax
cd $(pwd)

num=$1
cycle_num=$2

function float_gt() {
    perl -e "{if($1>$2){print 1} else {print 0}}"
}

mkdir picked
mkdir unpicked
mkdir finished

for i in $(cat batch${num})
do
name=align_${i}
name_flip=flip_align_${i}
echo $i

echo "*********************" >> log/logfile${num}.txt
echo Start aligning ${i} >> log/logfile${num}.txt

ChimeraX --nogui << EOF > log/temp_logfile${num}.txt
open /gpfs/home/ksheng/script/align_temp/L17all_round2from3_round3from1_round4from1_round4from1_250_r1.mrc
open $i
volume all step 1
fitmap #2 inMap #1 search $cycle_num
volume resample #2 onGrid #1
save $name #3
EOF

read cor1 rest<<< $(cat log/temp_logfile${num}.txt| grep correlation | grep -Eo '[+-]?[0-9]+([.][0-9]+)?') 

#echo compare is $(($(float_gt $cor1 0.9) == 1))
echo unflipped correlation is $cor1


if [ $(echo "${cor1}>0.88"| bc) -eq 1 ] ; then
echo $i >>log/pick${num}.log
mv $name picked/
mv $i finished/
echo pick unflipped structure alignment >> log/logfile${num}.txt
echo fit unflipped correlation: $cor1 >> log/logfile${num}.txt
echo $cor1>> log/pick${num}.log
continue
fi

rm $name
echo "unflipped alignment failed, start with flipping structure" >> log/logfile${num}.txt

ChimeraX --nogui << EOF > log/temp_flip_logfile${num}.txt
open /gpfs/home/ksheng/script/align_temp/L17all_round2from3_round3from1_round4from1_round4from1_250_r1.mrc
open $i
volume flip #2
volume all step 1
fitmap #3 inMap #1 search $cycle_num
volume resample #3 onGrid #1
save $name_flip #4
EOF


read cor2 rest<<< $(cat log/temp_flip_logfile${num}.txt| grep correlation | grep -Eo '[+-]?[0-9]+([.][0-9]+)?') 
echo flipped correlation is $cor2

if [ $(echo "${cor2}>0.88" | bc) -eq 1 ]; then
mv $name_flip picked/
mv $i finished/
echo $i >>log/pick${num}.log
echo pick flipped structure alignment >> log/logfile${num}.txt
echo fit flipped correatlion: $cor2 >> log/logfile${num}.txt
echo $cor2>> log/pick${num}.log
continue

fi
rm $name_flip

echo alignment failed >> log/logfile${num}.txt
#mv $i unpicked/

done

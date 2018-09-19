#!/bin/bash
# MAIN_DIR='/mnt/c/Users/ryanc/Documents/corpora/mgb_subset' ;
# MAIN_DIR='/mnt/Shared/people/ryan/fred-s' ; COPY_DIR='/mnt/Shared/people/ryan/fred-s_clean' ; find "$MAIN_DIR" -maxdepth 2 -type d -print0 | while IFS= read -rd '' dir ; do for wav_file in "$dir/"*.wav ; do org="${dir//.}/$(basename $wav_file)" ; new="${COPY_DIR}/$(basename $wav_file)" ; echo "$org"; echo "$new"; sox "$org" noise-audio.wav trim 0 0.900; sox noise-audio.wav -n noiseprof noise.prof; sox "$org" "$new" noisered noise.prof 0.21; rm noise.prof | rm noise-audio.wav ; done ; done
# MAIN_DIR='/mnt/Shared/people/ryan/fred-s' ; COPY_DIR='/mnt/Shared/people/ryan/fred-s_clean' ; find "$MAIN_DIR" -maxdepth 2 -type d -print0 | while IFS= read -rd '' dir ; do for wav_file in "$dir/"*.wav ; do new="${COPY_DIR}${wav_file#$MAIN_DIR}" ; echo "$wav_file"; echo "$new"; sox "$wav_file" noise-audio.wav trim 0 0.900; sox noise-audio.wav -n noiseprof noise.prof; sox "$wav_file" "$new" noisered noise.prof 0.21; rm noise.prof | rm noise-audio.wav ; done ; done

MAIN_DIR='/mnt/Shared/people/ryan/fred-s' ; 
COPY_DIR='/mnt/Shared/people/ryan/fred-s_clean' ; 
find "$MAIN_DIR" -maxdepth 2 -type d -print0 | while IFS= read -rd '' dir ; 
do 
	for wav_file in "$dir/"*.wav ; 
	do 
		new="${COPY_DIR}${wav_file#$MAIN_DIR}" ; 
		echo "$wav_file"; echo "$new"; 
		sox "$wav_file" noise-audio.wav trim 0 0.900; 
		sox noise-audio.wav -n noiseprof noise.prof; 
		sox "$wav_file" "$new" noisered noise.prof 0.21; 
		rm noise.prof | rm noise-audio.wav ; 
	done ; 
done
FILE=$1

if [[ $FILE != "base_gray_color" && $FILE != "base_gray_texture" &&  $FILE != "skeleton_gray_color" && $FILE != "skeleton_gray_texture" && $FILE != "average_skeleton" ]]; then
  echo "Available datasets are base_gray_color, base_gray_texture, skeleton_gray_color, skeleton_gray_texture and average skeleton"
  exit 1
fi

echo "Specified [$FILE]"

URL=http://59.108.48.27/dualnet_public_release/$FILE.zip
ZIP_FILE=./datasets/$FILE.zip
TARGET_DIR=./datasets/
wget -N $URL -O $ZIP_FILE
tar -zxvf $ZIP_FILE -C ./datasets/
rm $ZIP_FILE

FILE=$1

if [[ $FILE != "base_gray_color" && $FILE != "base_gray_texture" &&  $FILE != "skeleton_gray_color" && $FILE != "skeleton_gray_texture" ]]; then
  echo "Available datasets are base_gray_color, base_gray_texture, skeleton_gray_color and skeleton_gray_texture"
  exit 1
fi

echo "Specified [$FILE]"

URL=http://59.108.48.27/dualnet_public_release/$FILE.zip
ZIP_FILE=./datasets/$FILE.zip
TARGET_DIR=./datasets/
wget -N $URL -O $TAR_FILE
tar -zxvf $ZIP_FILE -C ./datasets/
rm $ZIP_FILE

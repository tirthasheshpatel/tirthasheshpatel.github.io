echo "statring..."

FILES=`find -O3 -L -name "*.md"`

j=0
for file in $FILES
do
	sed -i 's/\$.*\$/\\\\\( .* \\\\\)/g' $file
	j=$((j+1))
	echo "processed $j files..."
done

echo "successful"


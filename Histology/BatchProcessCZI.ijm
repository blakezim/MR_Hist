input = "/hdscratch/ucair/microscopic/18_061_Microscopic/czi/"
output = "/hdscratch/ucair/microscopic/18_061_Microscopic/tiff/"
setBatchMode(true);
count = 0;
dir = getDirectory(input);
countFiles(dir);
n = 0;
print(count+" files to processe ... ");
function mainImage(input, output, filename) { 
	// function description
	run("Bio-Formats", "open="+input+filename+" color_mode=Default display_metadata rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT use_virtual_stack series_1");
	wait(1000);
	whiteBalance();
	saveAs("Tiff", output + filename);
	close("*");
    close("Original Metadata - "+filename);
}
function labelImage(input, output, filename) { 
	// function description
	run("Bio-Formats Macro Extensions");
	Ext.setId(input+filename);
	Ext.getSeriesCount(seriesCount);
	lbl_id = seriesCount - 1;
	run("Bio-Formats", "open="+input+filename+" color_mode=Default display_metadata rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT use_virtual_stack series_"+lbl_id);
	run("Stack to RGB");
	selectWindow(filename+" - label image (RGB)");
	saveAs("Tiff", output + filename + "label");
	close("*");
    close("Original Metadata - "+filename);
}
function whiteBalance(){
	name=getTitle();
	run("Split Channels");
	MeanColor=newArray(3);
	maxi = 0;
	for (u=1; u<4; u++) {
		selectWindow("C"+u+"-"+name);
		// Type of region, dimensions and position can be changed.
		// A region in the backgroung (no object) should be choosen.
		makeRectangle(100,100,100,100);
		// The user can be prompted to draw a region at that stage. Remove the double slash in the next line.
		// waitForUser("Please draw a region in the background");
		getStatistics(area, mean);
		MeanColor[u-1] = mean;
	if (mean>=maxi) maxi = mean;
	}
	for (u=1; u<4; u++) {
		selectWindow("C"+u+"-"+name);
		run("Select None");
		run("Multiply...", "value="+maxi/MeanColor[u-1]);
	}
	//print("Merge Channels...", "c1=C1-"+name+" c2=C2-"+name+" c3=C3-"+name+" create");
	run("Merge Channels...", "c1=[C1-"+name+"] c2=[C2-"+name+"] c3=[C3-"+name+"] create");
	//run("Merge Channels...", "create");
	run("RGB Color");
	selectWindow(name); close();
}
function countFiles(dir) {
      list = getFileList(dir);
      for (i=0; i<list.length; i++) {
          if (endsWith(list[i], "/"))
              countFiles(""+dir+list[i]);
          else
          	  filename = list[i];
          	  print("open="+input+filename);
          	  mainImage(dir, output, list[i]);
          	  labelImage(dir, output, list[i]);
              count++;
      }
}
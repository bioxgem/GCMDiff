<?php

    /***** USAGE *****

    
    AIM: GENERATE CHECKMOL+PUBCHEM7+DRUG+RING MOIETY
    USE: php Exec_generate_fea.php file/dir file_name/dir_path 0-4

    HELP:
    0 All 
    1 checkmol
    2 pubchem section7
    3 drug
    4 ring

     ***** GENERATE CHECKMOL+PUBCHEM7+DRUG MOIETY *****/


    ini_set("memory_limit", -1);

    if(isset($argv[3])) {

        $input_type = $argv[1];
        $input_name = $argv[2];
        $fea_type   = $argv[3];
    }

    else {
        echo "\nUSE: php Exec_generate_fea.php file/dir file_name/dir_path 0-4\n\n";
        echo "HELP:\n\t0 All \n\t1 checkmol\n\t2 pubchem section7\n\t3 drug\n\t4 ring\n\n";
        exit;
    }


    if($input_type === "file") {
        $infile = file_get_contents($input_name);
        $mol_files  = explode("\n", trim($infile));
    }


    elseif($input_type === "dir")
        $mol_files = glob($input_name."/*");
    

    else
        exit;


    $input_folder = "./PDB_ligand/";    // input mol file dir
    $output = "./PDB_ligand_cm_pcfp7_drug.txt";     // output file name

    $Mol_files = Glob("{$input_folder}/*");
    //$infile = file_get_contents("./chembl_compound_list.txt");
    //$Mol_files = explode("\n", trim($infile));

    $prog_cm = "checkmol-latest-linux-x86_64";
    $prog_matchmol = "matchmol";

    $PCFP7_Moieties = Glob("./PCFP_Section7_mol/*.mol");    // pcfp7 moiety
    $DRUG_Moieties = Glob("./drug_moiety_mol/*.mol");       // drug moiety


    /**** OUTPUT STRING START ****/

    $output_string = "compound";
    
    foreach (Range(1, 204) as $index) {
        $key = "#" . Str_Pad($index, 3, "0", STR_PAD_LEFT);
        $output_string .= "\t$key";
    }

    foreach (Range(714, 904) as $index) {
        $key = "#" . Str_Pad($index, 3, "0", STR_PAD_LEFT);
        $output_string .= "\t$key";
    }

    $output_string .= "\n";

    /**** OUTPUT STRING END ****/
    

    
    foreach ($Mol_files as $mol_count => $mol_name) {

        $c_id = SubStr(BaseName($mol_name), 0, -4);
        echo $mol_count."/".count($Mol_files)."\t".$c_id."\n";

        /**** CHECKMOL START ****/
        $Checkmol_counts = Array();
        foreach (Range(1, 204) as $index) {
            $key = "#" . Str_Pad($index, 3, "0", STR_PAD_LEFT);
            $Checkmol_counts[$key] = 0;
        }

        $Output_cm = Array();
        $cmd = "./{$prog_cm} -p $mol_name";
        Exec($cmd, $Output_cm);
        foreach ($Output_cm as $output_cm_line) {
            $Pieces = Explode(":", $output_cm_line);
            $Pieces = Array_Map("Trim", $Pieces);
            $cm_id = $Pieces[0];
            $cm_count = $Pieces[1];

            // CONVERT TO 0, 1
            if($cm_count >= 1)
                $Checkmol_counts[$cm_id] = 1;
            else
                $Checkmol_counts[$cm_id] = 0;
        }
        $output_string .= $c_id."\t".Implode("\t", $Checkmol_counts);

        unset ($Output_cm);
        unset ($Checkmol_counts);
        /**** CHECKMOL END ****/


        /**** PUBCHEM START ****/
        foreach ($PCFP7_Moieties as $moiety) {

            $output_array = Array();
            $cmd = "./{$prog_matchmol} {$moiety} {$mol_name}";
            Exec($cmd, $output_array, $ret_val);
            
            if (StrStr($output_array[0], 'T'))
                $output_string .= "\t1";
            
            elseif (StrStr($output_array[0], 'F'))
                $output_string .= "\t0";

            else
                echo "Identify " . SubStr(BaseName($moiety, ".mol"), 5) . " failed.\n";

            unset ($output_array);
        }
        /**** PUBCHEM END ****/



        /**** DRUG MOIETY START ****/
        foreach ($DRUG_Moieties as $moiety) {

            $output_array = Array();
            $cmd = "./{$prog_matchmol} {$moiety} {$mol_name}";
            Exec($cmd, $output_array, $ret_val);
            
            if (StrStr($output_array[0], 'T'))
                $output_string .= "\t1";
            
            elseif (StrStr($output_array[0], 'F'))
                $output_string .= "\t0";

            else
                echo "Identify " . SubStr(BaseName($moiety, ".mol"), 5) . " failed.\n";

            unset ($output_array);
        }

        $output_string .= "\n";
        /**** DRUG MOIETY END ****/
    }


    File_Put_Contents($output, $output_string, LOCK_EX);
        
    

   

?>

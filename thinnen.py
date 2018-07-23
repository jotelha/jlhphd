from dump import dump

d = dump("1_SDS_on_AU_111_6x4x1_1ns_npt_with_restarts_nptProduction.dump")
#d.tselect.skip(10)
d.aselect.test("$type != 9 and $type != 10")  
d.write("1_SDS_on_AU_111_1ns_npt_200fs_steps_nonwater.dump")

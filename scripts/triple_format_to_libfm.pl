#!/usr/bin/perl -w
# Copyright (C) 2010, 2011, 2012, 2013, 2014 Steffen Rendle
# Contact:   srendle@libfm.org, http://www.libfm.org/
#
# This file is part of libFM.
#
# libFM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# libFM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with libFM.  If not, see <http://www.gnu.org/licenses/>.
#
#
# triple_format_to_libfm.pl: Converts data in a triple format
# "id1 id2 id3 target" (like often used in recommender systems for rating
# prediction) into the libfm format.
#
# Version history
# - 2013-07-12: write groups
# - 2012-12-27: header is not printed

use Getopt::Long;
use strict;

srand();


my $file_in;
my $file_out_meta;
my $has_header = 0;
my $target_column = undef;
my $_delete_column = "";
my $offset = 0; # where to start counting for indices. For libsvm one should start with 1; libfm can deal with 0.
my $separator = " ";

# example
# ./triple_format_to_libfm.pl --in train.txt,test.txt --header 0 --target_column 2 --delete_column 3,4,5,6,7 --offset 0


GetOptions(
	'in=s'             => \$file_in,
	'header=i'	   => \$has_header,
	'target_column=i'  => \$target_column,
	'delete_column=s'  => \$_delete_column,
	'offset=i'         => \$offset,
	'separator=s'      => \$separator,
	'outmeta=s'        => \$file_out_meta,
);

(defined $target_column) || die "no target column specified";

my @files = split(/[,;]/, $file_in);
my %delete_column;
foreach my $c (split(/[,;]/, $_delete_column)) {
	$delete_column{int($c)} = 1;
}

my %id;
my $id_cntr = $offset;

my $OUT_GROUPS;
if (defined $file_out_meta) {
	open $OUT_GROUPS, '>' , $file_out_meta;
}

foreach my $file_name (@files) {
	my $file_out = $file_name . ".libfm";
	print "transforming file $file_name to $file_out...";
	my $num_triples = 0;

	open my $IN, '<' , $file_name;
	open my $OUT, '>' , $file_out;
	if ($has_header) {
		$_ = <$IN>;
#		print {$OUT} $_;
	}
	while (<$IN>) {
		chomp;
		if ($_ ne "") {
			my @data = split /$separator/;
			($#data >= $target_column) || die "not enough values in line $num_triples, expected at least $target_column values\nfound $_\n";
			my $out_str = $data[$target_column];
			my $out_col_id = 0; ## says which column in the input a field corresponds to after "deleting" the "delete_column", i.e. it is a counter over the #$data-field in @data assuming that some of the columns have been deleted; one can see this as the "group" id
			for (my $i = 0; $i <= $#data; $i++) {
				if (($i != $target_column) && (! exists $delete_column{$i})) {
					my $col_id = $out_col_id . " " . $data[$i]; ## this id holds the unique id of $data[$i] (also w.r.t. its group)
					if (! exists $id{$col_id}) {
						$id{$col_id} = $id_cntr;
						if (defined $file_out_meta) {
							print {$OUT_GROUPS} $out_col_id, "\n";
						}
						$id_cntr++;
					}
					my $libfm_id = $id{$col_id};
					$out_str .= " " . $libfm_id . ":1";
					$out_col_id++;
				}
			}
			print {$OUT} $out_str, "\n";
		}
	}
	close $OUT;
	close $IN;
	print "\n";
}

if (defined $file_out_meta) {
	close $OUT_GROUPS;
}


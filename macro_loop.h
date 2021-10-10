
#define MACRO_LOOP(N, FUN) MACRO_LOOP##N(FUN)

/* Common Lisp code for generating the below boilerplate

(loop :for i :below 32
      :do (if (= i 0)
              (format t "#define MACRO_LOOP~A(FUN) FUN(~A);" (+ i 1) i)
              (format t "#define MACRO_LOOP~A(FUN) FUN(~A); MACRO_LOOP~A(FUN)"
                      (+ i 1) i i))
                      (terpri))

 */

#define MACRO_LOOP1(FUN) FUN(0);
#define MACRO_LOOP2(FUN) FUN(1); MACRO_LOOP1(FUN)
#define MACRO_LOOP3(FUN) FUN(2); MACRO_LOOP2(FUN)
#define MACRO_LOOP4(FUN) FUN(3); MACRO_LOOP3(FUN)
#define MACRO_LOOP5(FUN) FUN(4); MACRO_LOOP4(FUN)
#define MACRO_LOOP6(FUN) FUN(5); MACRO_LOOP5(FUN)
#define MACRO_LOOP7(FUN) FUN(6); MACRO_LOOP6(FUN)
#define MACRO_LOOP8(FUN) FUN(7); MACRO_LOOP7(FUN)
#define MACRO_LOOP9(FUN) FUN(8); MACRO_LOOP8(FUN)
#define MACRO_LOOP10(FUN) FUN(9); MACRO_LOOP9(FUN)
#define MACRO_LOOP11(FUN) FUN(10); MACRO_LOOP10(FUN)
#define MACRO_LOOP12(FUN) FUN(11); MACRO_LOOP11(FUN)
#define MACRO_LOOP13(FUN) FUN(12); MACRO_LOOP12(FUN)
#define MACRO_LOOP14(FUN) FUN(13); MACRO_LOOP13(FUN)
#define MACRO_LOOP15(FUN) FUN(14); MACRO_LOOP14(FUN)
#define MACRO_LOOP16(FUN) FUN(15); MACRO_LOOP15(FUN)
#define MACRO_LOOP17(FUN) FUN(16); MACRO_LOOP16(FUN)
#define MACRO_LOOP18(FUN) FUN(17); MACRO_LOOP17(FUN)
#define MACRO_LOOP19(FUN) FUN(18); MACRO_LOOP18(FUN)
#define MACRO_LOOP20(FUN) FUN(19); MACRO_LOOP19(FUN)
#define MACRO_LOOP21(FUN) FUN(20); MACRO_LOOP20(FUN)
#define MACRO_LOOP22(FUN) FUN(21); MACRO_LOOP21(FUN)
#define MACRO_LOOP23(FUN) FUN(22); MACRO_LOOP22(FUN)
#define MACRO_LOOP24(FUN) FUN(23); MACRO_LOOP23(FUN)
#define MACRO_LOOP25(FUN) FUN(24); MACRO_LOOP24(FUN)
#define MACRO_LOOP26(FUN) FUN(25); MACRO_LOOP25(FUN)
#define MACRO_LOOP27(FUN) FUN(26); MACRO_LOOP26(FUN)
#define MACRO_LOOP28(FUN) FUN(27); MACRO_LOOP27(FUN)
#define MACRO_LOOP29(FUN) FUN(28); MACRO_LOOP28(FUN)
#define MACRO_LOOP30(FUN) FUN(29); MACRO_LOOP29(FUN)
#define MACRO_LOOP31(FUN) FUN(30); MACRO_LOOP30(FUN)
#define MACRO_LOOP32(FUN) FUN(31); MACRO_LOOP31(FUN)
